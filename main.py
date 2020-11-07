import argparse
import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

import buildDataset
from Constants import Constants
from Regressor import Regressor
from Regressor_Manager import Regressor_Manager
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


def load_data():
    image_list = glob.glob('face_images/*.jpg')
    print("Length of given Image List", len(image_list))

    Utils().train_test_split()

    training_image_list = glob.glob('data/train/class/*.jpg')
    validation_image_list = glob.glob('data/val/class/*.jpg')

    print("Length of training Image List", len(training_image_list))
    print("Length of validation Image List", len(validation_image_list))


def build_dataset(cuda=False, num_workers=1):
    # define pytorch transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(128)
    ])

    train_datasets = []
    train_datasets.append(buildDataset.AugmentImageDataset('data/train'))

    for i in range(9):
        train_datasets.append(buildDataset.AugmentImageDataset('data/train', transform))
    augmented_dataset = ConcatDataset(train_datasets)
    print("Length of Augmented Dataset", len(augmented_dataset))

    train_loader_args = dict(shuffle=True,
                             batch_size=Constants.REGRESSOR_BATCH_SIZE_CUDA,
                             num_workers=num_workers, pin_memory=True) \
        if cuda else dict(shuffle=True, batch_size=Constants.REGRESSOR_BATCH_SIZE_CPU)
    augmented_dataset_batch = DataLoader(dataset=augmented_dataset, **train_loader_args)

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

    return augmented_dataset_batch


if __name__ == '__main__':
    device, is_cuda_present, num_workers = Utils.get_device()
    print("Device: {0}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    load_data()
    augmented_dataset_batch = build_dataset(is_cuda_present, num_workers)
    # print_util(augmented_dataset_batch)

    regressor_train_arguments = {
        "data_loader": augmented_dataset_batch,
        "saved_model_path": Constants.REGRESSOR_SAVED_MODEL_PATH,
        "epochs": Constants.REGRESSOR_EPOCH,
        "learning_rate": Constants.REGRESSOR_LR,
        "weight_decay": Constants.REGRESSOR_WEIGHT_DECAY,
        "in_channel": Constants.REGRESSOR_IN_CHANNEL,
        "hidden_channel": Constants.REGRESSOR_HIDDEN_CHANNEL,
        "out_dims": Constants.REGRESSOR_OUT_DIMS,
        "loss_plot_path": Constants.REGRESSOR_LOSS_PLOT_PATH
    }

    regressor_manager = Regressor_Manager()
    regressor_manager.train(regressor_train_arguments, device)
