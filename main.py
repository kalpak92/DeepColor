import argparse
import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

import buildDataset
from Colorize_deep import Colorize_deep
from Colorizer import Colorizer
from Constants import Constants
from Regressor import Regressor
from utils import Utils


def print_util(augmented_dataset_batch):
    sample = next(iter(augmented_dataset_batch))
    l_channel, a_channel, b_channel = sample

    print("L channel shape: ", l_channel.shape)
    print("a_channel shape:", a_channel.shape)
    print("b_channel shape:", b_channel.shape)

    regressor = Regressor(in_channel=1, hidden_channel=3, out_dims=2,
                          train_mode="regressor")
    output_hat = regressor(l_channel)
    print(output_hat.size())

    print(output_hat)

    print("------")
    a_channel_mean = a_channel.mean(dim=(2, 3))
    # print("a_channel_mean_size: ", a_channel_mean.size())
    # print("a_channel_mean: ", a_channel_mean)
    # print("------")
    b_channel_mean = b_channel.mean(dim=(2, 3))
    # print("b_channel_mean_size: ", b_channel_mean.size())
    # print("b_channel_mean: ", b_channel_mean)
    # print("-----")
    a_b_orig = torch.cat([a_channel_mean, b_channel_mean], dim=1)
    print("t_orig_size: ", a_b_orig.size())
    print("t_orig: ", a_b_orig)


def print_util_1(augmented_dataset_batch, activation_function):
    sample = next(iter(augmented_dataset_batch))
    l_channel, a_channel, b_channel = sample

    print("L channel shape: ", l_channel.shape)
    print("a_channel shape:", a_channel.shape)
    print("b_channel shape:", b_channel.shape)

    colorizer = Colorizer(in_channel=3, hidden_channel=3,
                          out_channel=2,
                          activation_function=activation_function)
    output_hat = colorizer(l_channel)
    print(output_hat.size())
    print(output_hat)
    print(a_channel)
    # Utils().show_img(torchvision.utils.make_grid(l_channel))
    # Utils().show_img(torchvision.utils.make_grid(a_channel))
    # Utils().show_img(torchvision.utils.make_grid(b_channel))


def load_data():
    image_list = glob.glob('face_images/*.jpg')
    print("Length of given Image List", len(image_list))

    Utils().train_test_split()

    training_image_list = glob.glob('data/train/class/*.jpg')
    validation_image_list = glob.glob('data/val/class/*.jpg')
    test_image_list = glob.glob('data/test/class/*.jpg')

    print("Length of training Image List", len(training_image_list))
    print("Length of validation Image List", len(validation_image_list))
    print("Length of testing Image List", len(test_image_list))


def build_dataset(cuda=False, num_workers=1,
                  activation_function=Constants.SIGMOID):
    # define pytorch transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(128)
    ])

    train_datasets = []
    if activation_function == Constants.SIGMOID or activation_function == Constants.TANH:
        train_datasets.append(buildDataset.AugmentImageDataset('data/train'))
    elif activation_function == Constants.RELU:
        train_datasets.append(buildDataset.AugmentImageDataset_RELU('data/train'))

    for i in range(9):
        if activation_function == Constants.SIGMOID:
            train_datasets.append(buildDataset.AugmentImageDataset('data/train', transform))
        elif activation_function == Constants.RELU:
            train_datasets.append(buildDataset.AugmentImageDataset_RELU('data/train', transform))
        elif activation_function == Constants.TANH:
            train_datasets.append(buildDataset.AugmentImageDataset_Tanh('data/train',
                                                                        transform))

    augmented_dataset = ConcatDataset(train_datasets)
    print("Length of Augmented Dataset", len(augmented_dataset))

    train_loader_args = dict(shuffle=True,
                             batch_size=Constants.REGRESSOR_BATCH_SIZE_CUDA,
                             num_workers=num_workers, pin_memory=True) \
        if cuda else dict(shuffle=True, batch_size=Constants.REGRESSOR_BATCH_SIZE_CPU)

    augmented_dataset_batch_train = DataLoader(dataset=augmented_dataset, **train_loader_args)
    augmented_dataset_batch_val = DataLoader(dataset=buildDataset.AugmentImageDataset('data/val'))
    augmented_dataset_batch_test = DataLoader(dataset=buildDataset.AugmentImageDataset('data/test'))

    # print(sample.size())
    # l_channel, a_channel, b_channel = sample
    # print("L channel shape: ", l_channel.shape)
    # print("a_channel shape:", a_channel.shape)
    # print("b_channel shape:", b_channel.shape)

    # current_image = torch.vstack((l_channel[0], a_channel[0], b_channel[0]))
    # print(current_image.shape)
    # # print("L: ", l_channel[0][0])
    # print("Sample: ",sample[0][0].shape)

    # utils.show_img(torchvision.utils.make_grid(l_channel))
    # utils.show_img(torchvision.utils.make_grid(a_channel))
    # utils.show_img(torchvision.utils.make_grid(b_channel))

    return augmented_dataset_batch_train, augmented_dataset_batch_val, augmented_dataset_batch_test


if __name__ == '__main__':
    activation_function = Constants.TANH

    device, is_cuda_present, num_workers = Utils.get_device()
    print("Device: {0}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    load_data()

    augmented_dataset_batch_train, augmented_dataset_batch_val, \
    augmented_dataset_batch_test = build_dataset(is_cuda_present, num_workers,
                                                 activation_function)
    # print_util_1(augmented_dataset_batch_test, activation_function)

    colorizer_deep = Colorize_deep()
    # colorizer_deep.train_regressor(augmented_dataset_batch_train, device)
    colorizer_deep.train_colorizer(augmented_dataset_batch_train, augmented_dataset_batch_val,
                                 activation_function, device)

    # colorizer_deep.test_colorizer(augmented_dataset_batch_test, device)
    # Utils.show_output_image("outputs/gray/Orig_img_10.jpg", "Gray")
    # Utils.show_output_image("outputs/color/Orig_img_10.jpg", "Original")
    # Utils.show_output_image("outputs/color/Recons_img_10.jpg", "Reconstructed")
