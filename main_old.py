import cv2 as cv2

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import lego_handler

# Helper libraries

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import os

from random import shuffle


# KERAS

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


train_data= 'brickdataset/train/'
test_data='brickdataset/valid/'

background_image = 'background.jpg'

# ONLY FOR TRANSPARENT DATASET
def remove_transparency(source, bg):
	image = source
	trans_mask = image[:,:,3] == 0
	image[trans_mask] = [bg,bg,bg,bg]
	new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	return new_img

dataset_id = [
	"2357 Brick corner 1x2x2",
"3003 Brick 2x2",
"3004 Brick 1x2",
"3005 Brick 1x1",
"3022 Plate 2x2",
"3023 Plate 1x2",
"3024 Plate 1x1",
"3040 Roof Tile 1x2x45deg",
"3069 Flat Tile 1x2",
"3673 Peg 2M",
"3713 Bush for Cross Axle",
"3794 Plate 1X2 with 1 Knob",
"6632 Technic Lever 3M",
"11214 Bush 3M friction with Cross axle",
"18651 Cross Axle 2M with Snap friction",
"32123 half Bush"
]

def load_

tr_lbl_data = np.load("train_label.npy")
tr_image_data = np.load("train_image.npy")
tst_lbl_data = np.load("test_label.npy")
tst_image_data = np.load("test_image.npy")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(tr_image_data, tr_lbl_data, epochs=77)


test_loss, test_acc = model.evaluate(tst_image_data, tst_lbl_data)

print('Test accuracy:', test_acc)



