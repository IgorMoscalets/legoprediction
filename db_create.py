import cv2 as cv2
# Helper libraries

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import os

from random import shuffle

train_data= 'myowndataset/'
test_data='myowndataset/'

background_image = 'background.jpg'

dataset_id = os.listdir(train_data)

BRIGHTNESS_BIAS = 90


BACKGROUND_VALUE = 180 + BRIGHTNESS_BIAS


RESOLUTION_X = 32


# INCREASE BRIGHTNESS
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v -= value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img

# CROP IMG FOR X AND Y TRAINING DATA

def crop_image_xy(img):

	xsize = 160
	# 1: x=222 y=126 2: x=420 y=309
	x1=222
	y1=126

	x2=x1+xsize
	y2=y1+xsize

	# X AND Y ARE FLIPPED
	crop_img = img[y1:y2,x1:x2]
	return crop_img

# ONLY FOR TRANSPARENT DATASET
def remove_transparency(source, bg):
	image = source
	trans_mask = image[:,:,3] == 0
	image[trans_mask] = [bg,bg,bg,bg]
	new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	return new_img

def train_data_label():
	image_check = 0
	train_images = []
	for j in range(len(dataset_id)):
		train_data_curr = train_data + dataset_id[j] + "/"
		for i in tqdm(os.listdir(train_data_curr)):
			path = os.path.join(train_data_curr, i)
			img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
			img = crop_image_xy(img)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (RESOLUTION_X, RESOLUTION_X))
			image_check = img
			
			train_images.append([np.array(img), j])
	shuffle(train_images)
	#cv2.imshow("image",image_check)
	#cv2.waitKey(0)
	return train_images


def test_data_label():
	test_images = []
	for j in range(len(dataset_id)):
		test_data_curr = test_data + dataset_id[j] + "/"
		for i in tqdm(os.listdir(test_data_curr)):
			path = os.path.join(test_data_curr, i)
			img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
			img = crop_image_xy(img)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (RESOLUTION_X, RESOLUTION_X))
			test_images.append([np.array(img), j])
	shuffle(test_images)
	return test_images


training_images = train_data_label()
testing_images = test_data_label()

tr_image_data = np.array([i[0] for i in training_images]) / 255.0 
#print tr_image_data
tr_lbl_data = np.array([i[1] for i in training_images])

tst_image_data = np.array([i[0] for i in testing_images]) / 255.0
tst_lbl_data = np.array([i[1] for i in testing_images])
#print training_images
#print tr_lbl_data

np.save("train_label.npy", tr_lbl_data)
np.save("train_image.npy", tr_image_data)
np.save("test_image.npy", tst_image_data)
np.save("test_label.npy", tst_lbl_data)
