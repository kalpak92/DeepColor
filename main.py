import glob

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

import buildDataset
from Colorize_deep import Colorize_deep
from Constants import Constants
from utils import Utils


def load_data():
    image_list = glob.glob('face_images/*.jpg')
    print("Length of given Image List", len(image_list))

    Utils().train_test_split()

    training_image_list = glob.glob('data/train/class/*.jpg')
    # validation_image_list = glob.glob('data/val/class/*.jpg')
    test_image_list = glob.glob('data/test/class/*.jpg')

    print("Length of training Image List", len(training_image_list))
    # print("Length of validation Image List", len(validation_image_list))
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
    # augmented_dataset_batch_val = DataLoader(dataset=buildDataset.AugmentImageDataset('data/val'))
    augmented_dataset_batch_test = DataLoader(dataset=buildDataset.AugmentImageDataset('data/test'))

    return augmented_dataset_batch_train, augmented_dataset_batch_test


def execute_colorizer_tanh():
    activation_function = Constants.TANH
    save_path = {'grayscale': 'outputs_tanh/gray/', 'colorized': 'outputs_tanh/color/'}
    device, is_cuda_present, num_workers = Utils.get_device()
    model_name = Constants.COLORIZER_SAVED_MODEL_PATH_TANH

    print("Device: {0}".format(device))
    augmented_dataset_batch_train, \
    augmented_dataset_batch_test = build_dataset(is_cuda_present, num_workers,
                                                 activation_function)

    colorizer_deep = Colorize_deep()
    colorizer_deep.train_colorizer(augmented_dataset_batch_train,
                                    activation_function, model_name, device)

    colorizer_deep.test_colorizer(augmented_dataset_batch_test, activation_function,
                                  save_path, model_name, device)

    colorizer_deep.train_regressor(augmented_dataset_batch_train, device)
    colorizer_deep.test_regressor(augmented_dataset_batch_test, device)


def execute_colorizer_sigmoid():
    activation_function = Constants.SIGMOID
    save_path = {'grayscale': 'outputs_sigmoid/gray/', 'colorized': 'outputs_sigmoid/color/'}
    device, is_cuda_present, num_workers = Utils.get_device()
    model_name = Constants.COLORIZER_SAVED_MODEL_PATH_SIGMOID

    print("Device: {0}".format(device))
    augmented_dataset_batch_train, \
    augmented_dataset_batch_test = build_dataset(is_cuda_present, num_workers,
                                                 activation_function)

    colorizer_deep = Colorize_deep()
    colorizer_deep.train_colorizer(augmented_dataset_batch_train,
                                   activation_function, model_name, device)

    colorizer_deep.test_colorizer(augmented_dataset_batch_test, activation_function,
                                  save_path, model_name, device)

    colorizer_deep.train_regressor(augmented_dataset_batch_train, device)
    colorizer_deep.test_regressor(augmented_dataset_batch_test, device)


if __name__ == '__main__':
    load_data()

    print("Normal Credit - Sigmoid")
    execute_colorizer_sigmoid()

    print("Extra Credit - Tanh")
    execute_colorizer_tanh()
