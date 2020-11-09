import numpy as np
import torch
from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets


class AugmentImageDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        global img_a, img_b, img_gray
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        original_image = np.asarray(img)

        img_lab = rgb2lab(original_image)
        img_lab = img_lab + 128
        img_lab = img_lab / 255

        img_a = img_lab[:, :, 1:2]
        img_a = torch.from_numpy(img_a.transpose((2, 0, 1))).float()  # To match the channel dimensions

        img_b = img_lab[:, :, 2:3]
        # plt.imshow(img_b)
        # plt.show()
        img_b = torch.from_numpy(img_b.transpose((2, 0, 1))).float()

        img_gray = rgb2gray(original_image)
        # plt.imshow(img_gray)
        # plt.show()
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_a, img_b


class AugmentImageDataset_Tanh(datasets.ImageFolder):
    def __getitem__(self, index):
        global img_a, img_b, img_gray
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        original_image = np.asarray(img)

        img_lab = rgb2lab(original_image)
        img_lab = img_lab + 128
        img_lab = img_lab / 255

        img_a = img_lab[:, :, 1:2]
        img_a = torch.from_numpy(img_a.transpose((2, 0, 1))).float()  # To match the channel dimensions

        img_b = img_lab[:, :, 2:3]
        # plt.imshow(img_b)
        # plt.show()
        img_b = torch.from_numpy(img_b.transpose((2, 0, 1))).float()

        img_gray = rgb2gray(original_image)
        # plt.imshow(img_gray)
        # plt.show()
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_a, img_b


class AugmentImageDataset_RELU(datasets.ImageFolder):
    def __getitem__(self, index):
        global img_a, img_b, img_gray
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        original_image = np.asarray(img)

        img_lab = rgb2lab(original_image)
        img_a = img_lab[:, :, 1:2]
        img_a = torch.from_numpy(img_a.transpose((2, 0, 1))).float()  # To match the channel dimensions

        img_b = img_lab[:, :, 2:3]
        # plt.imshow(img_b)
        # plt.show()
        img_b = torch.from_numpy(img_b.transpose((2, 0, 1))).float()

        img_gray = rgb2gray(original_image)
        # plt.imshow(img_gray)
        # plt.show()
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_a, img_b
