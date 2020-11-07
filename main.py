import argparse

import torch
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import buildDataset
import utils


def show_img(img):
    plt.figure(figsize=(18, 15))
    # unnormalize
    # img = img / 2 + 0.5
    # np_img = img.numpy()
    # np_img = np.clip(np_img, 0., 1.)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def load_data():
    image_list = glob.glob('face_images/*.jpg')
    print("Length of given Image List", len(image_list))

    utils.train_test_split()

    training_image_list = glob.glob('data/train/class/*.jpg')
    validation_image_list = glob.glob('data/val/class/*.jpg')

    print("Length of training Image List", len(training_image_list))
    print("Length of validation Image List", len(validation_image_list))


def build_dataset(cuda=False, num_workers=0):
    # define pytorch transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(128)
    ])

    train_datasets = [buildDataset.AugmentImageDataset('data/train')]
    for i in range(9):
        train_datasets.append(buildDataset.AugmentImageDataset('data/train', transform))
    augmented_dataset = ConcatDataset(train_datasets)
    print("Length of Augmented Dataset", len(augmented_dataset))

    train_loader_args = dict(shuffle=True, batch_size=256, num_workers=num_workers, pin_memory=True) if cuda \
        else dict(shuffle=True, batch_size=32)
    augmented_dataset_batch = DataLoader(dataset=augmented_dataset, **train_loader_args)
    sample = next(iter(augmented_dataset_batch))

    l_channel, a_channel, b_channel = sample
    print("L channel shape: ", l_channel.shape)
    print("a_channel shape:", a_channel.shape)
    print("b_channel shape:", b_channel.shape)

    # current_image = torch.vstack((l_channel[0], a_channel[0], b_channel[0]))
    # print(current_image.shape)
    # # print("L: ", l_channel[0][0])
    # print("Sample: ",sample[0][0].shape)


if __name__ == '__main__':
    cuda = False
    if torch.cuda.is_available():
        cuda = True
        device = 'cuda'
    else:
        device = 'cpu'

    num_workers = 8 if cuda else 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    load_data()
    build_dataset(cuda, num_workers)
