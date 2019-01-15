import torchvision.transforms as transforms
import numpy as np
import cv2
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


Tensor_to_Image = transforms.Compose([
    transforms.Normalize([0.0, 0.0, 0.0], [1.0/0.229, 1.0/0.224, 1.0/0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1.0, 1.0, 1.0]),
    transforms.ToPILImage()
])


def tensor_to_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


# de-preprocess and convert to tensorboard image
def tensor_to_X_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = np.transpose(image, (2, 0, 1))
    return image


colors = np.array([[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255],
                   [80, 128, 255],
                   [255, 230, 180],
                   [255, 0, 255],
                   [0, 255, 255],
                   [100, 0, 0],
                   [0, 100, 0],
                   [255, 255, 0],
                   [50, 150, 0],
                   [200, 255, 255],
                   [255, 200, 255],
                   [128, 128, 80],
                   # [0, 50, 128],
                   # [0, 100, 100],
                   [0, 255, 128],
                   [0, 128, 255],
                   [255, 0, 128],
                   [128, 0, 255],
                   [255, 128, 0],
                   [128, 255, 0],
                   [0, 0, 0]
                   ])


def apply_mask(image, mask, alpha=0.5, ignore_index=5):
    c, h, w = image.size()

    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    color_mask = colors[mask.reshape(-1)].reshape((h, w, 3))
    masked_image = image * alpha + color_mask * (1-alpha)

    # ignore index
    if not (ignore_index is None):
        mask = mask.reshape(h, w, 1)
        masked_image = masked_image * (mask != ignore_index) + image * (mask == ignore_index)

    masked_image = cv2.cvtColor(masked_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    masked_image = np.transpose(masked_image, (2, 0, 1))
    return masked_image
