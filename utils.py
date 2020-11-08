import os
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb
import matplotlib.image as mpimg

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
        os.makedirs('data/test/class/', exist_ok=True)
        os.makedirs('data/val/class', exist_ok=True)

        number_of_images = len(next(os.walk('face_images'))[2])
        print("Number of images - ", number_of_images)

        for i, file in enumerate(os.listdir('face_images')):
            if i < (0.1 * number_of_images):  # first 10% will be val
                copy2('face_images/' + file, 'data/test/class/' + file)
                continue
            elif i < (0.15 * number_of_images):
                copy2('face_images/' + file, 'data/val/class/' + file)
                continue
            else:  # others will be train
                copy2('face_images/' + file, 'data/train/class/' + file)

        print("Training Set Size : ", len(next(os.walk('data/train/class'))[2]))
        print("Validation Set Size : ", len(next(os.walk('data/val/class'))[2]))
        print("Test Set Size : ", len(next(os.walk('data/test/class'))[2]))

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

    @staticmethod
    def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
        plt.clf()
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
        color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
          plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
          plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


    @staticmethod
    def show_output_image(path, title):
        plt.clf()
        image = mpimg.imread(path)
        plt.title(title)
        plt.imshow(image)
        plt.show()
