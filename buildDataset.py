import torch
from skimage.color import rgb2lab, rgb2gray
import numpy as np
from torchvision import datasets


class AugmentImageDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        global img_ab, img_gray
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            original_image = self.transform(img)
            original_image = np.asarray(original_image)

            img_lab = rgb2lab(original_image)
            img_lab = (img_lab + 128) / 255

            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()      # To match the channel dimensions

            img_gray = rgb2gray(original_image)
            img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_ab, img_gray
