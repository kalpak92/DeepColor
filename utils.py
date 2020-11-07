from shutil import copy2
import os
import numpy as np
import matplotlib.pyplot as plt

def show_img(image):
    plt.figure(figsize=(20, 20))
    # image = image / 2 + 0.5
    np_img = image.numpy()
    #np_img = np.clip(np_img, 0, 1)
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

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
