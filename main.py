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


train_data= 'brickdataset/train/'
test_data='brickdataset/valid/'

background_image = 'background.jpg'

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


# ONLY FOR TRANSPARENT DATASET
def remove_transparency(source, bg):
	image = source
	trans_mask = image[:,:,3] == 0
	image[trans_mask] = [bg,bg,bg,bg]
	new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
	return new_img


def label_images(img):
	label = img

	image_id = dataset_id.index(label)

	np_array_id = [
	np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
	np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
	np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
	np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
	np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
	np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
	np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
	np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
	np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
	np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
	np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
	np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
	np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
	np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
	np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
	np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
	]

	return np_array_id[image_id]

def train_data_label():
	train_images = []
	for j in range(len(dataset_id)):
		train_data_curr = train_data + dataset_id[j] + "/"
		for i in tqdm(os.listdir(train_data_curr)):
			path = os.path.join(train_data_curr, i)
			img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
			img = remove_transparency(img, 220)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (64, 64))
			train_images.append([np.array(img), label_images(dataset_id[j])])
	shuffle(train_images)	
	return train_images


def test_data_label():
	test_images = []

	for j in range(len(dataset_id)):
		test_data_curr = test_data + dataset_id[j] + "/"
		for i in tqdm(os.listdir(test_data_curr)):
			path = os.path.join(test_data_curr, i)
			img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
			img = remove_transparency(img, 220)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(img, (64, 64))
			test_images.append([np.array(img), label_images(dataset_id[j])])
	shuffle(test_images)
	return test_images

training_images = train_data_label()
testing_images = test_data_label()

tr_image_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
tr_lbl_data = np.array([i[1] for i in training_images])

#print training_images
#print tr_lbl_data

n_input = 64*64   # input layer (64x64 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 16   # output layer (0-9 digits)

learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) 


weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # print loss and accuracy (per minibatch)
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))