import utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

def show_image(image):
    plt.figure(figsize=(20,20))
    image = image/2 + 0.5
    npimg = image.numpy()
    npimg = np.clip(npimg, 0, 1)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    train_dir = 'data/train/class'
    val_dir = 'data/train/class'

    utils.train_test_split()

    train_data_transform_scale = transforms.Compose([
        transforms.Resize(256)
    ])

    train_data = utils.ImageTransformation('data/train/', train_data_transform_scale)

    train_data_transform_augment = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(256)
    ])

    for i in range(10):
        train_data = torch.utils.data.ConcatDataset(
            [train_data, utils.ImageTransformation('data/train/', train_data_transform_augment)]
        )

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    print(len(train_data_loader))
    sample = next(iter(train_data_loader))
    l, ab, target = sample
    print(l.shape)
    print(ab.shape)
    print(target.shape)

    show_image(torchvision.utils.make_grid(l))


if __name__ == '__main__':
    main()
