import numpy as np
import os
import cv2
from src.config import DATA_DIR, CLASSES, BATCH_SIZE, IMG_SIZE
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf

from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')


IMG_PATH = os.path.join(DATA_DIR,'images')
MASK_PATH = os.path.join(DATA_DIR,'masks')

def load_image(image_path):

    """
    Load all images
    :param image_path: (str) Path to the image folder
    :return: (194,IMG_SIZE,IMG_SIZE,3)

    """
    image_name = [f for f in os.listdir(image_path) if f.endswith(".jpg") and not f.startswith(".")]
    instances = []
    for f in image_name:
        img = cv2.imread(os.path.join(image_path,"{}".format(f)), cv2.COLOR_BGR2RGB)
        # NEED TO CHECK IF I CAN RESIZE
        # WOULD IT AFFECT THE MASK?
        img_resize = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        # CHANGE HERE IF NOT RESIZE
        instances.append(img_resize)
    images = np.stack(instances,axis=0)
    print("Images array shape: ",images.shape)
    return images

def load_mask(image_path, mask_path):
    """
    Load all masks for all images
    :param image_path: (str) path to image folder
    :param mask_path: (str) path to masks folder
    :return: (194,3,IMG_SIZE,IMG_SIZE)
    """
    image_name = [f for f in os.listdir(image_path) if f.endswith(".jpg") and not f.startswith(".")]
    subdir_name = [os.path.join(mask_path,f) for f in os.listdir(mask_path) if not f.startswith(".")][::-1]
    subdir_files = []
    class_mask_id = {k:i+1 for i,k in enumerate(CLASSES)}
    # Get the name of files in the subdirectories
    for dir in subdir_name:
        subdir_files.append([f for f in os.listdir(dir) if f.endswith("jpg") and not f.startswith(".")])

    all_mask = []

    for img in image_name:
        mask = []
        # Check for each class if we have a mask for that image
        for i,subdir in enumerate(subdir_name):
            # If the image has a mask
            if img in subdir_files[i]:
                m = cv2.imread(os.path.join(subdir,img),0)
                m = cv2.resize(m,(IMG_SIZE,IMG_SIZE))
                mask.append(np.where(m>0,1,0))
            else:
                mask.append(np.zeros((IMG_SIZE,IMG_SIZE)))
        # We have 3 separate masks
        mask = np.stack(mask,axis=-1)
        all_mask.append(mask)

    all_mask = np.array(all_mask)
    print("All masks shape: ",all_mask.shape)

    return all_mask

def show_image(ind,img_arr,mask_arr):
    """
    Function to visualize image and check if the masks match
    :param ind: (int) index of the image in imageto visualize
    :param img_arr: (np array) image array of shape (194,IMG_SIZE,IMG_SIZE,3). Obtained from load_image()
    :param mask_arr: (np array) mask array of shape (194,3,IMG_SIZE,IMG_SIZE). Obtained from load_mask()
    :return: visualization of images and masks
    """
    print("Showing image")
    plt.imshow(cv2.cvtColor(img_arr[ind,:,:,:], cv2.COLOR_BGR2RGB))
    plt.show()
    for m in mask_arr[ind]:
        print("Mask is empty: ", (m==0).all())
        plt.imshow(m,'gray',vmin=0,vmax=1)
        plt.show()


def load_data(image_path, mask_path, valid_size=0.1,test_size=0.1):
    """

    :param image_path: (str) path to folder data/images
    :param mask_path: (str) path to folder data/masks
    :param valid_size: (float) percentage of data used for validation
    :param test_size: (float) percentage of data used for testing
    :return: (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    train_x: (np array) shape (num_train,IMG_SIZE,IMG_SIZE,3)
    train_y: (np array) shape (num_train,3,IMG_SIZE,IMG_SIZE)
    """
    image_name = [f for f in os.listdir(image_path) if f.endswith(".jpg") and not f.startswith(".")]
    total_size = len(image_name)
    valid_size = int(valid_size * total_size)
    test_size = int(test_size * total_size)

    images = load_image(image_path)
    masks = load_mask(image_path,mask_path)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=2020)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=2020)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=2020)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=2020)

    print("Train shape: ", train_x.shape, train_y.shape)
    print("Valid shape: ", valid_x.shape, valid_y.shape)
    print("Test shape: ", test_x.shape, test_y.shape)


    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def tf_dataset(x, y, batch=8):
    """
    :param x: (numpy array) Image of shape (n,IMG_SIZE,IMG_SIZE,3)
    :param y: (numpy array) Mask of shape (n,3,IMG_SIZE,IMG_SIZE)
    :param batch: (int) BATCH_SIZE from default_config
    :return: Tensorflow dataset pipeline, divided into batches
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

