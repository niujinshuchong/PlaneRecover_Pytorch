import torch
from torch.utils import data
import torchvision.transforms as tf
import  torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import argparse
from PIL import Image
from scipy.io import loadmat

from net import PlanePredNet
from utils import AverageMeter, tensor_to_image

torch.manual_seed(123)
np.random.seed(123)


class PlaneDataset(data.Dataset):
    def __init__(self, txt_file='train_8000.txt', transform=None, root_dir=None, resize=False):
        self.transform = transform
        self.root_dir = root_dir
        self.txt_file = os.path.join(self.root_dir, txt_file)
        self.resize = resize

        self.data_list = [line.strip().replace(' ', '/') for line in open(self.txt_file, 'r').readlines()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prefix = os.path.join(self.root_dir, self.data_list[index])
        image = cv2.imread(prefix + '.jpg')
        depth = cv2.imread(prefix + '_depth.png', -1)
        label = cv2.imread(prefix + '_label.png', -1)
         
        cam = np.array(list(map(float, open(prefix + '_cam.txt').readline().strip().split(',')))).reshape(3, 3)

        # calculate all K_inv for different scale output
        K_invs = []
        for s in range(4):
            C = cam.copy()
            C[0, 0] = C[0, 0] / (2**s)
            C[1, 1] = C[1, 1] / (2**s)
            C[0, 2] = C[0, 2] / (2**s)
            C[1, 2] = C[1, 2] / (2**s)
            K_inv = np.linalg.inv(C)
            K_invs.append(K_inv)
        K_invs = np.stack(K_invs)

        h, w, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        sample = {
            'image' : image, 
            'depth' : torch.Tensor(depth[:, :, 0] / 100.).view(1, h, w),
            'label' : torch.Tensor(label).view(1, h, w),
            'K_inv' : torch.Tensor(K_invs)
        }

        return sample


def generate_homogeneous(h=192, w=320):
    x = torch.arange(w, dtype=torch.float).view(1, w)
    y = torch.arange(h, dtype=torch.float).view(h, 1)

    x = x.cuda()
    y = y.cuda()
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)
    xy1 = torch.stack((xx, yy, torch.ones((h, w)).cuda()))   # (3, h, w)
    xy1 = xy1.view(3, -1)                                    # (3, h*w)

    return xy1


def get_plane_parameters(plane_parameters):
    # infer plane depth
    plane_depth = torch.norm(plane_parameters, 2, dim=1, keepdim=True)  # (n, 1)

    # infer plane normal and depth
    plane_normal = plane_parameters / plane_depth    # (n, 3)
    
    return plane_depth, plane_normal


def plot(image, parameters, embeddings, depth, label):
    h, w, _ = image.shape
    prediction = embeddings.detach().numpy().argmax(axis=1).reshape(h, w)

    cv2.imshow("image", image)
    cv2.imshow("pred", prediction.astype(np.uint8)*50)
    cv2.waitKey(0)


def load_dataset(subset):
    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if subset == 'train':
        dataset = PlaneDataset(txt_file='train_8000.txt', transform=transforms, root_dir='data')
        loaders = data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    else:
        dataset = PlaneDataset(txt_file='tst_100.txt', transform=transforms, root_dir='data')
        loaders = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    return loaders


def get_loss(plane_params, pred_masks, depth, label, K_inv):

    mask_loss, depth_loss = 0., 0.
    for scale, pred_mask in zip(range(len(pred_masks)),
                                pred_masks):
        b, c, h, w = pred_mask.size()

        # downsample 
        cur_depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)
        #cur_label = F.interpolate(label, (h, w), mode='nearest')
        cur_label = F.interpolate(label, (h, w), mode='bilinear', align_corners=False)
        # assume K_inv for different images is same
        cur_K_inv = K_inv[0, scale].clone()          

        # infer all Q for all pixel
        xy1 = generate_homogeneous(h, w)
        ray = torch.matmul(cur_K_inv, xy1)                # (3, h*w)
        Q = ray.unsqueeze(0) * cur_depth.view(b, 1, h*w)      # (b, 3, h*w)

        diff = torch.abs(torch.matmul(plane_params, Q) - 1.)

        logits = pred_mask.view(b, -1, h*w)
        prob = torch.nn.functional.softmax(logits, dim=1) # (b, c, h*w)
        diff = diff * prob[:, :-1, :]

        depth_loss += torch.sum(torch.mean(torch.mean(diff, dim=2), dim=0))

        # mask loss
        plane_prob = prob[:,:-1, :].sum(dim=1, keepdim=True)
        non_plane_prob = prob[:,-1:, :]
        cur_label = cur_label.view(b, 1, -1)
        mask_loss += torch.mean(-cur_label*torch.log(plane_prob+1e-8) - (1.0 - cur_label)*torch.log(non_plane_prob+1e-8))

    loss = 0.1*mask_loss + depth_loss
    return loss, mask_loss, depth_loss

def train(net, optimizer, data_loader, epoch):
    net.train()
    losses = AverageMeter()
    losses_mask = AverageMeter()
    losses_depth = AverageMeter()

    for iter, sample in enumerate(data_loader):
        image = sample['image'].cuda()
        depth = sample['depth'].cuda()
        label = sample['label'].cuda()
        K_inv = sample['K_inv'].cuda()

        # forward
        plane_params, pred_masks = net(image)

        # loss
        loss, loss_mask, loss_depth = get_loss(plane_params, pred_masks, depth, label, K_inv)     

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        losses_mask.update(loss_mask.item())
        losses_depth.update(loss_depth.item())

        if iter % 10 == 0:
            print(f"[{epoch:2d}][{iter:4d}/{len(data_loader)}]"
                  f"Loss:{losses.val:.4f} ({losses.avg:.4f})"
                  f"Mask:{losses_mask.val:.4f} ({losses_mask.avg:.4f})"
                  f"Depth:{losses_depth.val:.4f} ({losses_depth.avg:.4f})")

def eval(net, data_loader):
    net.eval()
    for iter, sample in enumerate(data_loader):
        image = sample['image'].cuda()
        with torch.no_grad():
            params, pred_masks = net(image)

        image = tensor_to_image(image[0].cpu())       
        logits = pred_masks[1][0].cpu().numpy()
        prediction = logits.argmax(axis=0)
        print(len(np.unique(prediction)))
        params = params[0]
        norm = params.norm(dim=1, keepdim=True)
        params = params / norm
        print(torch.cat((params, 1./norm), dim=1))
        for i in range(6):
            print(np.sum(prediction == i))
        predictions = []
        for i in range(6):
            predictions.append(prediction == i)
        prediction = np.concatenate(predictions, axis=0)
        cv2.imshow("image", image)
        cv2.imshow("prediction", prediction.astype(np.uint8)*250)
        #cv2.imwrite("prediction"+str(iter)+'.png', prediction.astype(np.uint8)*50)
        cv2.waitKey(0)


def main():
    args = parse_args()

    net = PlanePredNet(5)
    net.cuda()

    data_loader = load_dataset(args.mode)

    if args.resume_dir is not None:
        model_dict = torch.load(args.resume_dir)
        net.load_state_dict(model_dict)

    if args.mode == 'eval':
        eval(net, data_loader)
        exit(0)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=0.0001, weight_decay=0.00001)

    checkpoint_dir = os.path.join('experiments', args.train_id, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(300):
        train(net, optimizer, data_loader, epoch)
        if epoch % 10 == 0:
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                    help='mode',
                    required=True)
    parser.add_argument('--resume_dir', type=str,
                    help='where to resume model for evaluation',
                    required=False)
    parser.add_argument('--train_id', type=str,
                    help='train id for training',
                    required=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()

