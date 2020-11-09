import glob
import os
from shutil import copy2

import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb
import matplotlib.image as mpimg


class Utils:
    @staticmethod
    def get_hyperparameters():
        parameters = dict(
            lr=[.001, 0.0001],
            weight_decay=[1e-5, 1e-6],
            epoch=[100, 200]
        )
        hyperparam_values = [v for v in parameters.values()]
        return hyperparam_values

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
    def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None, display_path=None, display_name=None):
        plt.clf()
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()     # combine channels
        color_image = color_image.transpose((1, 2, 0))                      # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
            plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
            plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))

        if display_path is not None and display_name is not None:
            os.makedirs('display/gray/', exist_ok=True)
            os.makedirs('display/color/', exist_ok=True)
            os.makedirs('display/reconstructed_color/', exist_ok=True)

            plt.imsave(arr=grayscale_input, fname='{}{}'.format(display_path['grayscale'], display_name['grayscale']), cmap='gray')
            plt.imsave(arr=color_image, fname='{}{}'.format(display_path['color'], display_name['color']))

    @staticmethod
    def display_result():
        images = []
        image_path_gray = glob.glob(r"display/gray/*.jpg")
        image_path_color = glob.glob(r"display/color/*.jpg")
        image_path_reconstructed = glob.glob(r"display/reconstructed_color/*jpg")
        images_path = zip(image_path_gray, image_path_color, image_path_reconstructed)

        for image_path in images_path:
            image = Image.open(image_path[0]).convert('RGB')
            image = np.array(image)
            images.append(image)
            image = Image.open(image_path[1]).convert('RGB')
            image = np.array(image)
            images.append(image)
            image = Image.open(image_path[2]).convert('RGB')
            image = np.array(image)
            images.append(image)

        images = np.array(images)
        titles = ["GrayScale Image", "Original Image", "Reconstructed Image"]
        Utils.show_multiple_images(images, cols=10, titles=titles*10)
        # Utils.show_img(images)



    @staticmethod
    def show_output_image(path, title):
        plt.clf()
        image = mpimg.imread(path)
        plt.title(title)
        plt.imshow(image)
        plt.show()


    def show_multiple_images(images, cols=1, titles=None):
        """Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        plt.clf()
        assert ((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.savefig("plot.png")
