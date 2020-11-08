import os
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
import torch


class Utils:
    @staticmethod
    def show_img(image):
        plt.figure(figsize=(20, 20))
        # image = image / 2 + 0.5
        np_img = image.numpy()
        # np_img = np.clip(np_img, 0, 1)
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    @staticmethod
    def train_test_split():
        os.makedirs('data/train/class/', exist_ok=True)
        os.makedirs('data/val/class/', exist_ok=True)

        number_of_images = len(next(os.walk('face_images'))[2])
        print("Number of images - ", number_of_images)

        for i, file in enumerate(os.listdir('face_images')):
            if i < (0.1 * number_of_images):  # first 10% will be val
                copy2('face_images/' + file, 'data/val/class/' + file)
            else:  # others will be train
                copy2('face_images/' + file, 'data/train/class/' + file)

        print("Validation Set Size : ", len(next(os.walk('data/val/class'))[2]))
        print("Training Set Size : ", len(next(os.walk('data/train/class'))[2]))

    # display(Image(filename='data/train/class/image00007.jpg'))

    @staticmethod
    def get_device():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_cuda_present = True if torch.cuda.is_available() else False
        num_workers = 8 if is_cuda_present else 0

        return device, is_cuda_present, num_workers

    @staticmethod
    def get_ab_mean(a_channel, b_channel):
        a_channel_mean = a_channel.mean(dim=(2, 3))
        b_channel_mean = b_channel.mean(dim=(2, 3))
        a_b_mean = torch.cat([a_channel_mean,
                              b_channel_mean], dim=1)
        return a_b_mean

    @staticmethod
    def plot_loss_epoch(train_loss_avg, fig_name):
        plt.ion()
        fig = plt.figure()
        plt.plot(train_loss_avg)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.show()
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()

    @staticmethod
    def show_img_tensor(image):
        plt.figure(figsize=(20, 20))
        plt.imshow(image.permute(1, 2, 0))
        plt.show()
        plt.clf()
