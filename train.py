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
import random

from net import PlanePredNet
from utils import AverageMeter, tensor_to_image, tensor_to_X_image, apply_mask

from path import Path
from tensorboardX import SummaryWriter

torch.manual_seed(123)
np.random.seed(123)


class PlaneDataset(data.Dataset):
    def __init__(self, txt_file='train_8000.txt', transform=None, root_dir=None, resize=False, mode='train'):
        self.transform = transform
        self.root_dir = root_dir
        self.txt_file = os.path.join(self.root_dir, txt_file)
        self.resize = resize
        self.mode = mode
        self.data_list = [line.strip().replace(' ', '/') for line in open(self.txt_file, 'r').readlines()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prefix = os.path.join(self.root_dir, self.data_list[index])
        image = cv2.imread(prefix + '.jpg')
        depth = cv2.imread(prefix + '_depth.png', -1)
        label = cv2.imread(prefix + '_label.png', -1)

        cam = np.array(list(map(float, open(prefix + '_cam.txt').readline().strip().split(',')))).reshape(3, 3)

        # add random flip
        if random.random() > 0.5 and self.mode == 'train':
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

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


def gaussian_smooth(x):
    # Create gaussian kernels
    kernel = torch.FloatTensor([[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]])
    kernel = kernel.t() * kernel
    kernel = kernel.view(1, 1, 7, 7).cuda()

    # Apply smoothing
    x_smooth = F.conv2d(x, kernel, padding=3)

    return x_smooth


def depth_2_normal(depth, cur_K_inv):
    b, c, h, w = depth.size()
    assert (c == 1)

    #depth = gaussian_smooth(depth)

    # infer all Q for all pixel
    xy1 = generate_homogeneous(h, w)
    ray = torch.matmul(cur_K_inv, xy1)              # (3, h*w)
    Q = ray.unsqueeze(0) * depth.view(b, 1, h * w)  # (b, 3, h*w)
    point3D = Q.view(b, 3, h, w)

    dx = point3D[:, :, :, 1:] - point3D[:, :, :, :-1]
    dy = point3D[:, :, :-1, :] - point3D[:, :, 1:, :]

    dx = dx[:, :, 1:, :]
    dy = dy[:, :, :, :-1]
    assert (dx.size() == dy.size())

    normal = torch.cross(dx, dy, dim=1)
    assert (normal.size() == dx.size())

    normal /= (torch.norm(normal, p=2, dim=1, keepdim=True) + 1e-6)
    #normal = (normal + 1) / 2.
    return normal


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
        dataset = PlaneDataset(txt_file='tst_100.txt', transform=transforms, root_dir='data', mode='val')
        loaders = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    return loaders


def get_loss(plane_params, pred_masks, depth, normal, label, K_inv):
    mask_loss, depth_loss, normal_loss = 0., 0., 0.
    for scale, pred_mask in zip(range(len(pred_masks)),
                                pred_masks):
        b, c, h, w = pred_mask.size()

        # downsample 
        cur_depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)
        cur_normal = F.interpolate(normal, (h, w), mode='bilinear', align_corners=False)

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

        # normal loss,  param is of size (b, num, 3)
        '''
        weight = prob[:, :-1, :] / (plane_prob + 1e-6)
        plane_normal = plane_params / (torch.norm(plane_params, dim=2, keepdim=True) + 1e-6 ) # (b, num, 3)
        plane_normal = plane_normal.permute(0, 2, 1)  # (b, 3, num)
        weighted_pixel_normal = torch.matmul(plane_normal, weight)   # (b, 3, h*w)
        sim = F.cosine_similarity(weighted_pixel_normal, cur_normal.view(b, 3, -1), dim=1)
        normal_loss += torch.mean((1. - sim) * cur_label.view(b, -1))
        '''

        plane_normal = plane_params / (torch.norm(plane_params, dim=2, keepdim=True) + 1e-6 ) # (b, num, 3)
        normal_diff = 1. - torch.matmul(plane_normal, cur_normal.view(b, 3, -1)) # (b, num, h*w)
        normal_diff = normal_diff * prob[:, :-1, :]
        normal_loss += torch.sum(torch.mean(torch.mean(normal_diff, dim=2), dim=0))

    loss = 0.2*mask_loss + depth_loss + normal_loss

    return loss, mask_loss, depth_loss, normal_loss


def train(net, optimizer, data_loader, epoch, writer):
    net.train()
    losses = AverageMeter()
    losses_mask = AverageMeter()
    losses_depth = AverageMeter()
    losses_normal = AverageMeter()

    for iter, sample in enumerate(data_loader):
        image = sample['image'].cuda()
        depth = sample['depth'].cuda()
        label = sample['label'].cuda()
        K_inv = sample['K_inv'].cuda()

        # normal
        normal = depth_2_normal(depth, K_inv[0, 0])

        # forward
        plane_params, pred_masks = net(image)

        # loss
        loss, loss_mask, loss_depth, loss_normal = get_loss(plane_params, pred_masks, depth, normal, label, K_inv)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        losses_mask.update(loss_mask.item())
        losses_depth.update(loss_depth.item())
        losses_normal.update(loss_normal.item())

        if iter % 10 == 0:
            print(f"[{epoch:2d}][{iter:4d}/{len(data_loader)}]"
                  f"Loss:{losses.val:.4f} ({losses.avg:.4f})"
                  f"Mask:{losses_mask.val:.4f} ({losses_mask.avg:.4f})"
                  f"Depth:{losses_depth.val:.4f} ({losses_depth.avg:.4f})"
                  f"Normal:{losses_normal.val:.4f} ({losses_normal.avg:.4f})")
            writer.add_scalar('loss/total_loss', losses.val, iter + epoch * len(data_loader))
            writer.add_scalar('loss/mask_loss', losses_mask.val, iter + epoch * len(data_loader))
            writer.add_scalar('loss/depth_loss', losses_depth.val, iter + epoch * len(data_loader))
            writer.add_scalar('loss/normal_loss', losses_normal.val, iter + epoch * len(data_loader))

        if iter % 100 == 0:
            mask = F.softmax(pred_masks[0], dim=1)
            normal = (normal + 1) / 2.
            for j in range(image.size(0)):
                writer.add_image('Train Input Image/%d'%(j), tensor_to_X_image(image[j].cpu()), iter + epoch * len(data_loader))
                writer.add_image('Train GT Depth/%d'%(j), 1. / depth[j], iter + epoch * len(data_loader))
                writer.add_image('Train GT Mask/%d'%(j), label[j], iter + epoch * len(data_loader))

                # apply mask to input image
                cur_mask = mask[j].detach().cpu().numpy().argmax(axis=0)
                masked_image = apply_mask(image[j].cpu(), cur_mask, ignore_index=mask.size(1) - 1)
                writer.add_image('Train Masked Image/%d' % (j), masked_image, iter + epoch * len(data_loader))

                # predict mask
                for k in range(mask.size(1) - 1):
                    writer.add_image('Train Mask %d/%d'%(k, j), mask[j, k:k+1], iter + epoch * len(data_loader))

                # non plane mask
                writer.add_image('Train Non-plane Mask/%d'%(j), mask[j, -1:], iter + epoch * len(data_loader))

                writer.add_image('Train Normal/%d'%(j), normal[j], iter + epoch * len(data_loader))


def eval(net, data_loader, epoch, writer):
    print('Evaluatin at epoch %d'%(epoch))
    net.eval()
    for iter, sample in enumerate(data_loader):
        image = sample['image'].cuda()
        depth = sample['depth'].cuda()
        label = sample['label'].cuda()

        with torch.no_grad():
            params, pred_masks = net(image)

        mask = F.softmax(pred_masks[0], dim=1)
        for j in range(image.size(0)):
            writer.add_image('Val Input Image/%d' % (j+iter*image.size(0)), tensor_to_X_image(image[j].cpu()),
                             iter + epoch * len(data_loader))
            writer.add_image('Val GT Depth/%d' % (j+iter*image.size(0)), 1. / depth[j], iter + epoch * len(data_loader))
            writer.add_image('Val GT Mask/%d' % (j+iter*image.size(0)), label[j], iter + epoch * len(data_loader))

            # apply mask to input image
            cur_mask = mask[j].detach().cpu().numpy().argmax(axis=0)
            masked_image = apply_mask(image[j].cpu(), cur_mask, ignore_index=mask.size(1) - 1)
            writer.add_image('Val Masked Image/%d' % (j+iter*image.size(0)), masked_image, iter + epoch * len(data_loader))

            # predict mask
            for k in range(mask.size(1) - 1):
                writer.add_image('Val Mask %d/%d' % (k, j+iter*image.size(0)), mask[j, k:k + 1], iter + epoch * len(data_loader))

            # non plane mask
            writer.add_image('Val Non-plane Mask/%d' % (j+iter*image.size(0)), mask[j, -1:], iter + epoch * len(data_loader))


def main():
    args = parse_args()

    net = PlanePredNet(args.plane_num)
    net.cuda()

    train_loader = load_dataset('train')
    val_loader = load_dataset('val')

    if args.resume_dir is not None:
        model_dict = torch.load(args.resume_dir)
        net.load_state_dict(model_dict)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=0.0001, weight_decay=0.00001)

    save_path = Path(os.path.join('experiments', args.train_id))
    checkpoint_dir = save_path/'checkpoints'
    checkpoint_dir.makedirs_p()

    # tensorboard writer
    writer = SummaryWriter(save_path)

    #eval(net, val_loader, -1, writer)

    for epoch in range(args.epochs):
        train(net, optimizer, train_loader, epoch, writer)

        if epoch % 10 == 0:
            eval(net, val_loader, epoch, writer)
            torch.save(net.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plane_num', default=5, type=int,
                        help='total training epochs',
                        required=False)
    parser.add_argument('--resume_dir', type=str,
                    help='where to resume model for evaluation',
                    required=False)
    parser.add_argument('--train_id', type=str,
                    help='train id for training',
                    required=False)
    parser.add_argument('--epochs', default=300, type=int,
                        help='total training epochs',
                        required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()

