from shutil import copy2
import os


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
