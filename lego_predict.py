import cv2 as cv2

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

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


train_data= 'myowndataset/'
test_data='myowndataset/'

background_image = 'background.jpg'

resolution_x = 32



dataset_id = os.listdir(train_data)


tr_lbl_data = np.load("train_label.npy")
tr_image_data = np.load("train_image.npy")
tst_lbl_data = np.load("test_label.npy")
tst_image_data = np.load("test_image.npy")

# print tst_image_data

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(resolution_x, resolution_x)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(tr_image_data, tr_lbl_data, epochs=140)


test_loss, test_acc = model.evaluate(tst_image_data, tst_lbl_data)

print('Test accuracy:', test_acc)


lowerBound=np.array([0,190,40])
upperBound=np.array([179,255,255])

cam= cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 0.5
fontColor              = (255,255,255)
lineType               = 1


while True:
	ret, img=cam.read()

	xsize = 160


	# 1: x=222 y=126 2: x=420 y=309
	x1=222
	y1=126

	x2=x1+xsize
	y2=y1+xsize

	cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255), 1)

	# X AND Y ARE FLIPPED
	crop_img = img[y1:y2,x1:x2]

	## PREDICTION SETTING
	crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	crop_img = cv2.resize(crop_img, (resolution_x, resolution_x))
	# crop_img = cv2.bitwise_not(crop_img) # INVERSION
	cv2.imshow("cropped", crop_img)

	check_img = []

	check_img.append(np.array([crop_img, 2]))
	check_image_data = np.array([i[0] for i in check_img]) / 255.0 


	# 	print check_image_data

	model_out = model.predict(check_image_data)
	#print model_out[0]

	prediction = np.argmax(model_out[0])

	prediction_answer = dataset_id[prediction]

	#print prediction_answer

	cv2.putText(img, prediction_answer, 
    (x2,y2-10), 
    font, 
    fontScale,
    fontColor,
    lineType)
			
	cv2.imshow("cam",img)
	cv2.waitKey(10)
