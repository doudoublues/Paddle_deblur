import random
from unittest.mock import patch

import numpy as np
import cv2
import random
from paddle import uint8
from paddle.vision.transforms import ColorJitter


def horizontal_flip(im):
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im

def vertical_flip(im):
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


class Compose:

    def __init__(self, transforms, to_rgb=True, patch_size=256):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.patch_size = patch_size
        

    def __call__(self, im1, im2):

        if isinstance(im1, str):
            im1 = cv2.imread(im1).astype('float32')
        if isinstance(im2, str):
            im2 = cv2.imread(im2).astype('float32')
        if im1 is None or im2 is None:
            raise ValueError('Can\'t read The image file {} and {}!'.format(im1, im2))
        if self.to_rgb:
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        h, w, c = im1.shape
        
        im1 = np.pad(im1, ((0, 512 - h), (0, 512 - w), (0, 0)), "reflect")
        im2 = np.pad(im2, ((0, 512 - h), (0, 512 - w), (0, 0)), "reflect")

        for op in self.transforms:
            outputs = op(im1, im2)
            im1 = outputs[0]
            im2 = outputs[1]

        # im2_s2 = resize(im2, self.patch_size // 2, cv2.INTER_CUBIC)
        # im2_s3 = resize(im2, self.patch_size // 4, cv2.INTER_CUBIC)

        im1 = np.transpose(im1, (2, 0, 1))
        im2 = np.transpose(im2, (2, 0, 1))
        # im2_s2 = np.transpose(im2_s2, (2, 0, 1))
        # im2_s3 = np.transpose(im2_s3, (2, 0, 1))

        return (im1, im2)

class Crop:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size
    def __call__(self, im1, im2):
        h, w, c = im1.shape
        patch_h = random.randint(0, h - self.patch_size)
        patch_w = random.randint(0, w - self.patch_size)
        im1 = im1[patch_h : patch_h + self.patch_size, patch_w : patch_w + self.patch_size, :]
        im2 = im2[patch_h : patch_h + self.patch_size, patch_w : patch_w + self.patch_size, :]
        return im1.astype(np.uint8), im2

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im1, im2):
        if random.random() < self.prob:
            im1 = horizontal_flip(im1)
            im2 = horizontal_flip(im2)
        return im1, im2

class ColorJitter2:
    def __init__(self, prob=0.5, bright=0.3, con = 0.3):
        self.prob = prob
        self.cj = ColorJitter(bright, con)
    def __call__(self, im1, im2):
        if random.random() < self.prob:
            im1 = self.cj._apply_image(im1)
            
        return im1, im2


class RandomVerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im1, im2):
        if random.random() < self.prob:
            im1 = vertical_flip(im1)
            im2 = vertical_flip(im2)
        return im1, im2
    
class RandomTranspose:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im1, im2):
        if random.random() < self.prob:
            im1 = im1.transpose(1,0,2)
            im2 = im2.transpose(1,0,2)
        return im1, im2




def normalize(im, mean, std):
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


class Normalize:

    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, (list, tuple))
                and isinstance(self.std, (list, tuple))):
            raise ValueError(
                "{}: input type is invalid. It should be list or tuple".format(
                    self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, im1, im2):

        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im1 = normalize(im1, mean, std)
        im2 = normalize(im2, mean, std)

        return im1, im2


class Resize:

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=(512, 512), interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("`interp` should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of `target_size` is invalid. It should be list or tuple, but it is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im1, im2):

        if not isinstance(im1, np.ndarray) or not (im2, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im1.shape) != 3 or len(im2.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im1 = resize(im1, self.target_size, self.interp_dict[interp])

        im2 = resize(im2, self.target_size, self.interp_dict[interp])

        return im1, im2

