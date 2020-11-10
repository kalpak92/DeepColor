import os
import re
from shutil import copy2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb

from Constants import Constants


class Utils:
    @staticmethod
    def get_hyperparameters():
        parameters = dict(
            lr=[.001],
            weight_decay=[1e-5],
            epoch=[100]
        )
        # parameters = dict(
        #     lr=[.0001],
        #     weight_decay=[1e-5],
        #     epoch=[100]
        # )
        hyperparams_values = [v for v in parameters.values()]
        return hyperparams_values

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
        # os.makedirs('data/val/class', exist_ok=True)

        number_of_images = len(next(os.walk('face_images'))[2])
        print("Number of images - ", number_of_images)

        for i, file in enumerate(os.listdir('face_images')):
            if i < (0.1 * number_of_images):  # first 10% will be val
                copy2('face_images/' + file, 'data/test/class/' + file)
                continue
            else:  # others will be train
                copy2('face_images/' + file, 'data/train/class/' + file)

        print("Training Set Size : ", len(next(os.walk('data/train/class'))[2]))
        # print("Validation Set Size : ", len(next(os.walk('data/val/class'))[2]))
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
    def to_rgb(grayscale_input, ab_input, activation_function=Constants.TANH,
               save_path=None, save_name=None, device="cpu"):
        plt.clf()
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()  # combine channels
        color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
        # print(color_image)
        # print(color_image.min())
        # if activation_function==Constants.TANH and bool(re.match('recons', save_name, re.I)):
        #     # mean = torch.mean(color_image)
        #     # std = torch.std()
        #     color_image = (color_image + 0.4) /0.4

        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
            plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
            plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))

    @staticmethod
    def show_output_image(gray, orig, recons, fig_name):
        plt.clf()
        f = plt.figure()
        f.add_subplot(1, 3, 1)
        plt.imshow(mpimg.imread(gray))
        plt.axis('off')
        f.add_subplot(1, 3, 2)
        plt.imshow(mpimg.imread(orig))
        plt.axis('off')
        f.add_subplot(1, 3, 3)
        plt.imshow(mpimg.imread(recons))
        plt.axis('off')

        # plt.show(block=True)

        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()
        plt.close()
        # image = mpimg.imread(path)
        # # plt.title(title)
        # plt.imshow(image)
        # plt.show()


class EarlyStopping_DCN:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0,
                 model_path=None,
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = model_path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ...')
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss
