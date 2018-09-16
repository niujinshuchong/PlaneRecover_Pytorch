import torch
from torch.utils import data
import torchvision.transforms as tf

import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
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
        K_inv = np.linalg.inv(np.array(cam))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        sample = {
            'image' : image, 
            'depth' : torch.Tensor(depth[:, :, 0] / 100.),
            'label' : torch.Tensor(label),
            'K_inv' : torch.Tensor(K_inv)
        }

        return sample


def generate_homogeneous(h=192, w=320):
    x = torch.arange(w).view(1, w)
    y = torch.arange(h).view(h, 1)

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

def get_loss(plane_params, logits, depth, label, K_inv):
    b, c, h, w = logits.size()
    # infer all depth
    xy1 = generate_homogeneous(h, w)
    ray = torch.matmul(K_inv[0], xy1)                # (3, h*w)
    Q = ray.unsqueeze(0) * depth.view(b, 1, h*w)     # (b, 3, h*w)

    diff = torch.abs(torch.matmul(plane_params, Q) - 1.)

    logits = logits.view(b, -1, h*w)
    prob = torch.nn.functional.softmax(logits, dim=1) # (b, c, h*w)
    # multiply label to ignore non planar region
    diff = diff * prob * label.view(b, 1, -1)

    return torch.mean(torch.sum(diff, dim=1))    

def train(net, optimizer, data_loader, epoch):
    net.train()
    losses = AverageMeter()

    for iter, sample in enumerate(data_loader):
        image = sample['image'].cuda()
        depth = sample['depth'].cuda()
        label = sample['label'].cuda()
        K_inv = sample['K_inv'].cuda()

        # forward
        plane_params, logits = net(image)

        # loss
        loss = get_loss(plane_params, logits, depth, label, K_inv)     

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())

        if iter % 10 == 0:
            print(f"[{epoch:2d}][{iter:4d}/{len(data_loader)}]"
                  f"Loss:{losses.val:.4f} ({losses.avg:.4f})")

def eval(net, data_loader):
    net.eval()
    for iter, sample in enumerate(data_loader):
        image = sample['image'].cuda()
        with torch.no_grad():
            _, logits = net(image)

        image = tensor_to_image(image[0].cpu())       
        logits = logits[0].cpu().numpy()
        prediction = logits.argmax(axis=0)
        cv2.imshow("image", image)
        cv2.imshow("prediction", prediction.astype(np.uint8)*60)
        cv2.waitKey(0)


def main(mode):
    net = PlanePredNet(5)
    net.cuda()

    data_loader = load_dataset(mode)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=0.0001, weight_decay=0.00001)

    checkpoint_dir = os.path.join('experiments', '1', 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if mode == 'eval':
        resume_dir = os.path.join(checkpoint_dir, f"network_epoch_268.pt")
        model_dict = torch.load(resume_dir)
        net.load_state_dict(model_dict)
        eval(net, data_loader)
        exit(0)

    for epoch in range(300):
        train(net, optimizer, data_loader, epoch)
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, f"network_epoch_{epoch}.pt"))


#main('train')
main('eval')

