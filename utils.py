from shutil import copy2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2gray, lab2rgb
from IPython.display import Image, display

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms


class ImageTransformation(datasets.ImageFolder):
    """Custom images folder, which converts images to grayscale before loading"""
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)

            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255

            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target


def train_test_split():
    os.makedirs('data/train/class/', exist_ok=True)
    os.makedirs('data/val/class/', exist_ok=True)

    number_of_images = len(next(os.walk('face_images'))[2])
    print("Number of images - ", number_of_images)

    for i, file in enumerate(os.listdir('face_images')):
        if i < (0.1 * number_of_images):  # first 10% will be val
            copy2('face_images/' + file, 'data/val/class/' + file)
            # os.rename('face_images/' + file, 'data/val/class/' + file)
        else:  # others will be train
            copy2('face_images/' + file, 'data/train/class/' + file)
            # os.rename('face_images/' + file, 'data/train/class/' + file)

    print("Validation Set Size : ", len(next(os.walk('data/val/class'))[2]))
    print("Training Set Size : ", len(next(os.walk('data/train/class'))[2]))

    display(Image(filename='data/train/class/image00007.jpg'))
