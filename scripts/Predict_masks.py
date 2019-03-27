import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TEST_PATH = '../input/stage1_test/'

def main():
    X_test, sizes_test = preprocess(
        IMG_WIDTH,
        IMG_HEIGHT,
        IMG_CHANNELS,
        TEST_PATH)

    model = load_model(
        '../output/model-dsbowl2018-1.h5',
        custom_objects={'dice_coef': dice_coef})
        
    preds = make_prediction(model, X_test, sizes_test)
    show_prediction(preds, X_test, n=5)


def preprocess(img_width, img_height, img_channels, test_path):
    test_ids = next(os.walk(test_path))[1]
    X_test = np.zeros(
        (len(test_ids), img_height, img_width, img_channels),
        dtype=np.uint8)

    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = test_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:img_channels]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_test[n] = img

    return X_test, sizes_test


def dice_coef(y_true, y_pred, T=0.5):
    y_pred_ = tf.to_float(y_pred > T)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred_)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))


def make_prediction(model, X_test, sizes_test):
    preds_test = model.predict(X_test, verbose=1)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(
            resize(np.squeeze(preds_test[i]), 
            (sizes_test[i][0], sizes_test[i][1]), 
            mode='constant', preserve_range=True))

    return preds_test_t


def show_prediction(preds, X_test, n=1):
    for _ in range(n):
        ix = random.randint(0, len(preds))
        plt.subplot(121)
        imshow(X_test[ix])
        plt.subplot(122)
        imshow(np.squeeze(preds[ix]))
        plt.show()


if __name__ == '__main__':
    main()