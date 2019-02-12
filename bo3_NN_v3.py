# -*- coding: utf-8 -*-
"""
Main update from v2:
	- tried to improve loading batches as this is what is slow by (see Batch_test.csv):
		0) have the photos unzipped, this is 5-10 times faster
		1) load big barils of images in RAM (~5000, or how many fit)
		2) take batches from there until used all images
		3) load the next big baril, etc.
"""

#Load libraries
import zipfile as zf
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

time_start = time.time()
np.set_printoptions(precision=0, suppress=True)
tf.reset_default_graph()

########################################################################################################
### Locating the files and setting the first parameters
########################################################################################################

onDesktop = True
onDesktop_onSSD = True
unzipped = True

originalSet = False
extended = False

use_cross_validation_from_train_set = False
use_test = True

load_model_and_predict_for_submission = True
filename_load_model = 'model_20190106033346.ckpt'

### Set The files
file = 'train_labels_extended20181223112328'
file_test = 'train_labels_extended20181223112328_balanced_disjoint_test_set'
original_size_images = (512,512)


### Set The folders
#path_data_FOLDER = r'C:\Users\Bogdan\Desktop\dataProtein'
if onDesktop:
	path_data_FOLDER = r'F:\MLdata\HumanProtein'
	path_save_model = r'F:\MLdata\HumanProtein'
	path_load_model = r'F:\MLdata\HumanProtein'
	if onDesktop_onSSD:
		path_data_FOLDER = r'C:\MLdata\HAP'
else:
	path_data_FOLDER = os.path.join(os.getcwd(), 'smalldata')
	path_save_model = r'C:\Users\Bogdan\Desktop\dataProtein'
	path_load_model = r'C:\Users\Bogdan\Desktop\dataProtein'
	if originalSet:
		path_data_FOLDER = r'C:\Users\Bogdan\Desktop\dataProtein'

if originalSet:
	file = 'train'
	file_test = 'train' # because the test file does not have labels
	original_size_images = (512,512)

if use_cross_validation_from_train_set:
	file_test = file

### Set The Channels & Nbr of Labels
channels = ['yellow', 'red', 'green', 'blue']
grad_methods = ['SGD', 'Adam', 'Adagrad']
nbr_classes = 28

### Loading all of this
zipname = file + '.zip'
labelname = file + '.csv'
zipname_test = file_test + '.zip'
labelname_test = file_test + '.csv'

if extended:
	labelname = 'train_labels_extended20181218132835.csv'
	labelname_test = labelname

if load_model_and_predict_for_submission:
	zipname_test = os.path.join(r'F:\MLdata\HumanProtein', 'test.zip')
	labelname_test = os.path.join(r'F:\MLdata\HumanProtein', 'sample_submission.csv')
	original_size_images = (512,512)

y_df = pd.read_csv(os.path.join(path_data_FOLDER, labelname))
nbr_training_exs = y_df['Id'].size

if 'ReflectRotate' not in y_df.columns:
	y_df['ReflectRotate'] = 0
	print('Addded reflec rotate column')

y_df = y_df.sample(frac=1).reset_index(drop=True) # just shuffling the dataset


########################################################################################################
### Some preliminary functions for the CNN
########################################################################################################

def split_into_train_test(Y_df, nbr_test_exs):
	return Y_df[:Y_df['Id'].size - nbr_test_exs], Y_df[Y_df['Id'].size - nbr_test_exs:].reset_index(drop = True)

''' The data set is extremely unbalanced, some classes have thousands of labels, while some have < 20.
A big value close to 1 will have the underrepresented classes to have big weights,
and thus a huge gradient when they are misclassified, adding bumps to the training error curve.
A small value close to 0 smoothess out the training error curve, but the algorithm won't learn much about
classifiying these underrepresented classes, probably classifying them wrong. '''
def calc_smoothing_weights(Y_df_train, smoothing_weights_val, use_weights):
	# Create weights to counter the unbalance of the data set
	label_weights_val = pd.DataFrame()
	# pass into one-hot encoding, for calculating the weights
	for i in range(nbr_classes):
		label_weights_val[i] = Y_df_train['Target'].map( lambda x: 1 if str(i) in x.strip().split() else 0)

	label_weights_val = label_weights_val.sum(axis=0).sort_values(ascending=False).sort_index().values
	#print_label_weigths_val = np.array([(idx, val, val / nbr_training_exs ) for idx, val in enumerate(label_weights_val.tolist())])
	#print('The distribution of labels is {}'.format(print_label_weigths_val))
	if smoothing_weights_val == 0 or use_weights == False:
		label_weights_val = np.ones_like([label_weights_val])
		return label_weights_val
	else:
		label_weights_val = np.array([1.0 / label_weights_val]) # inverse proportional to the distribution
		label_weights_val = label_weights_val / label_weights_val.min() # normalize to have smallest value = 1
		label_weights_val = np.power(label_weights_val, smoothing_weights_val) # smooth out the values by taking a root

	print('The max weight of the weight vector is {:2f}'.format(label_weights_val.max()))
	return label_weights_val

def load_image_from_zip(zipname, filename, new_size_images, reflecrotate):

	image = Image.open(zipname.open(filename)).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS)

	if reflecrotate == 0:
		return image
	elif reflecrotate == 1:
		return image.transpose(Image.FLIP_LEFT_RIGHT)
	elif reflecrotate == 2:
		return image.transpose(Image.FLIP_TOP_BOTTOM)
	elif reflecrotate == 3:
		return image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
	elif reflecrotate == 4:
		return image.rotate(90)
	elif reflecrotate == 5:
		return image.rotate(180)
	elif reflecrotate == 6:
		return image.rotate(270)

	return image

def load_image(filename, new_size_images, reflecrotate):

	image = Image.open(os.path.join(path_data_FOLDER, 'train', filename)).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS)

	if reflecrotate == 0:
		img = np.array(image)
	elif reflecrotate == 1:
		img = np.array(image.transpose(Image.FLIP_LEFT_RIGHT))
	elif reflecrotate == 2:
		img = np.array(image.transpose(Image.FLIP_TOP_BOTTOM))
	elif reflecrotate == 3:
		img = np.array(image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT))
	elif reflecrotate == 4:
		img = np.array(image.rotate(90))
	elif reflecrotate == 5:
		img = np.array(image.rotate(180))
	elif reflecrotate == 6:
		img = np.array(image.rotate(270))

	image.close()
	return img

def extract_batch(Y_df, zipname, new_size_images, batch_size, batch_nbr, X_pixels_mean):
	nbr_exs = Y_df['Id'].size
	global channels_to_consider

	if batch_size > nbr_exs:
		print('The batch size {} is larger than the total number of examples {}...'.format(batch_size, nbr_exs))

	batch_start = batch_nbr*batch_size % nbr_exs
	batch_end = (batch_nbr + 1)*batch_size % nbr_exs
	
	X = np.empty((batch_size, new_size_images[0], new_size_images[1], len(channels_to_consider)), dtype=np.uint8)
	
	if unzipped:
		if batch_start < batch_end:
			# pick the labels
			Y_str = Y_df['Target'].values[batch_start : batch_end]
			# load the images
			for j in range(batch_size):
				for i in range(len(channels_to_consider)):
					img = load_image(Y_df['Id'][batch_start + j] + '_' + channels_to_consider[i] + '.png', new_size_images, Y_df['ReflectRotate'][batch_start + j] )

					try:
						img = img - X_pixels_mean.reshape((new_size_images[0], new_size_images[1], len(channels_to_consider))).transpose(2,0,1)[i]
					except:
						print('The size of the images and the size of the pixel_mean do not match... ERROR')
						
					X[j].transpose(2,0,1)[i] = img
		elif batch_end == 0:
			# pick the labels
			Y_str = Y_df['Target'].values[batch_start :]
			# load the images
			for j in range(batch_size):
				for i in range(len(channels_to_consider)):
					img = load_image(Y_df['Id'][batch_start + j] + '_' + channels_to_consider[i] + '.png', new_size_images, Y_df['ReflectRotate'][batch_start + j] )
					
					try:
						img = img - X_pixels_mean.reshape((new_size_images[0], new_size_images[1], len(channels_to_consider))).transpose(2,0,1)[i]
					except:
						print('The size of the images and the size of the pixel_mean do not match... ERROR')
					
					X[j].transpose(2,0,1)[i] = img
			# shuffle the dataset as we went through it all
			Y_df = Y_df.sample(frac=1).reset_index(drop=True)
		else:
			print(' \nThis is the end of an epoch...')
			# pick the labels until we reach the end of the dataset
			Y1 = Y_df['Target'].values[batch_start: ]
			# laod the images until we reach the end of the dataset
			for j in range(nbr_exs - batch_start):
				for i in range(len(channels_to_consider)):
					img = load_image(Y_df['Id'][batch_start + j] + '_' + channels_to_consider[i] + '.png', new_size_images, Y_df['ReflectRotate'][batch_start + j] )
					
					try:
						img = img - X_pixels_mean.reshape((new_size_images[0], new_size_images[1], len(channels_to_consider))).transpose(2,0,1)[i]
					except:
						print('The size of the images and the size of the pixel_mean do not match... ERROR')
					
					X[j].transpose(2,0,1)[i] = img
			# shuffle the dataset as we went through it all
			Y_df = Y_df.sample(frac=1).reset_index(drop=True)
			# pick the remaining labels
			Y2 = Y_df['Target'].values[: batch_end]
			Y_str = np.append(Y1,Y2)
			# load the remaining images
			for j in range(batch_end):
				for i in range(len(channels_to_consider)):
					img = load_image(Y_df['Id'][j] + '_' + channels_to_consider[i] + '.png', new_size_images, Y_df['ReflectRotate'][j] )
					
					try:
						img = img - X_pixels_mean.reshape((new_size_images[0], new_size_images[1], len(channels_to_consider))).transpose(2,0,1)[i]
					except:
						print('The size of the images and the size of the pixel_mean do not match... ERROR')
					
					X[nbr_exs - batch_start + j].transpose(2,0,1)[i] = img
	else:
		print('USING UZIPPED STUFF, DEPRECATED, NO PIXEL MEAN, ETC')
		with zf.ZipFile(os.path.join(path_data_FOLDER, zipname), 'r') as zip_train:
			if batch_start < batch_end:
				Y_str = Y_df['Target'].values[batch_start : batch_end]

				X = np.empty((batch_size, new_size_images[0], new_size_images[1], len(channels_to_consider)), dtype=np.uint8)	
				for j in range(batch_size):
					for i in range(len(channels_to_consider)):
						img = load_image_from_zip(zip_train, Y_df['Id'][batch_start + j] + '_' + channels_to_consider[i] + '.png', new_size_images, Y_df['ReflectRotate'][batch_start + j] )
						X[j].transpose(2,0,1)[i] = img
			elif batch_end == 0:
				Y_str = Y_df['Target'].values[batch_start :]
				X = np.array([ np.stack([Image.open(zip_train.open(name+ '_' + color + '.png')).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS) for color in channels_to_consider], axis = -1) for name in Y_df['Id'].values[batch_start: ]])
				# we used all entries, shuffle for the next batch
				Y_df = Y_df.sample(frac=1).reset_index(drop=True)
			else:
				Y1 = Y_df['Target'].values[batch_start: ]
				X1 = np.array([ np.stack([Image.open(zip_train.open(name+ '_' + color + '.png')).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS) for color in channels_to_consider], axis = -1) for name in Y_df['Id'].values[batch_start: ]])
				# we used all entries, shuffle and get the remaining entries (if batch_size does not divide nbr_training_exs)
				Y_df = Y_df.sample(frac=1).reset_index(drop=True)
				Y2 = Y_df['Target'].values[: batch_end]
				X2 = np.array([ np.stack([Image.open(zip_train.open(name+ '_' + color + '.png')).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS) for color in channels_to_consider], axis = -1) for name in Y_df['Id'].values[: batch_end]])
				Y_str = np.append(Y1,Y2)
				X = np.append(X1,X2, axis=0)
				
    #create one-hot encoded output bor current batch
	Y = np.array([  [ int(str(i) in Y_str[j].split(' ')) for i in range(nbr_classes)  ] for j in range(batch_size)])
	return [X,Y]

def extract_batch_pred_test(Y_df_test, zipname, new_size_images, batch_size, batch_nbr, X_pixels_mean):
	nbr_exs = Y_df_test['Id'].size
	global channels_to_consider

	batch_start = batch_nbr*batch_size % nbr_exs
	batch_end = (batch_nbr + 1)*batch_size % nbr_exs
	
	if unzipped:
		X = np.empty((batch_size, new_size_images[0], new_size_images[1], len(channels_to_consider)), dtype=np.uint8)
		for j in range(batch_size):
			for i in range(len(channels_to_consider)):
				img = Image.open(os.path.join(path_data_FOLDER, 'test', Y_df_test['Id'][batch_start + j] + '_' + channels_to_consider[i] + '.png')).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS)
				
				try:
						img = img - X_pixels_mean.reshape((new_size_images[0], new_size_images[1], len(channels_to_consider))).transpose(2,0,1)[i]
				except:
					print('The size of the images and the size of the pixel_mean do not match... ERROR')
				
				X[j].transpose(2,0,1)[i] = img
	else:
		print('USING UZIPPED STUFF, DEPRECATED, NO PIXEL MEAN, ETC')
		with zf.ZipFile(os.path.join(path_data_FOLDER, zipname), 'r') as zip_test:
			X = np.array([ np.stack([Image.open(zip_test.open(name+ '_' + color + '.png')).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS) for color in channels_to_consider], axis = -1) for name in Y_df_test['Id'].values[batch_start:batch_end]])

	return X

def conv_layer(input, nbr_layer, strides = (1,1)):
	conv_layer = tf.layers.conv2d(input, conv_filter_nbrs[nbr_layer], (conv_filter_sizes[nbr_layer][0], conv_filter_sizes[nbr_layer][1]), \
				activation= leaky_relu, padding='same', strides = strides, \
				kernel_initializer = tf.initializers.truncated_normal(mean = 0, stddev = 0.1), \
				bias_initializer = tf.constant_initializer(0.1), \
				kernel_regularizer = tf.contrib.layers.l2_regularizer(beta), name='conv_layer_' + str(nbr_layer+1))
	if conv_pooling[nbr_layer]:
		return tf.layers.max_pooling2d(conv_layer, pool_size = (2,2), strides = 2, padding='same', name='conv_pool_layer_' + str(nbr_layer+1))
	else:
		return conv_layer

def full_layer(input, nbr_layer):
	full_layer = tf.layers.dense(input, full_nbr_neurons[nbr_layer], activation= leaky_relu, \
							kernel_initializer = tf.initializers.truncated_normal(mean = 0, stddev = 0.1), \
							bias_initializer = tf.constant_initializer(0.1), \
							kernel_regularizer = tf.contrib.layers.l2_regularizer(beta), name='full_layer_' + str(nbr_layer+1))
	if dropout_full_layer:
		return tf.layers.dropout(full_layer,rate=keep_prob, name='full_droppout_' + str(nbr_layer+1))
	else:
		return full_layer

########################################################################################################
### The hyperparameters about the architecture & Training of the NN
########################################################################################################

# Conv layers
nbr_conv_layers = 9
conv_filter_sizes = np.empty(shape = (nbr_conv_layers,2), dtype = np.uint8)
conv_filter_nbrs = np.empty(nbr_conv_layers, dtype = np.uint16)
conv_filter_strides = np.empty(shape = (nbr_conv_layers,2), dtype = np.uint8)
conv_pooling = np.empty(nbr_conv_layers, dtype = np.bool)

conv_filter_sizes[0] = (3,3)
conv_filter_strides[0] = (1,1)
conv_filter_nbrs[0] = 64
conv_pooling[0] = False

conv_filter_sizes[1] = (3,3)
conv_filter_strides[1] = (1,1)
conv_filter_nbrs[1] = 64
conv_pooling[1] = False

conv_filter_sizes[2] = (3,3)
conv_filter_strides[2] = (1,1)
conv_filter_nbrs[2] = 64
conv_pooling[2] = True

conv_filter_sizes[3] = (3,3)
conv_filter_strides[3] = (1,1)
conv_filter_nbrs[3] = 128
conv_pooling[3] = False

conv_filter_sizes[4] = (3,3)
conv_filter_strides[4] = (1,1)
conv_filter_nbrs[4] = 128
conv_pooling[4] = False

conv_filter_sizes[5] = (3,3)
conv_filter_strides[5] = (1,1)
conv_filter_nbrs[5] = 128
conv_pooling[5] = True

conv_filter_sizes[6] = (3,3)
conv_filter_strides[6] = (1,1)
conv_filter_nbrs[6] = 256
conv_pooling[6] = False

conv_filter_sizes[7] = (3,3)
conv_filter_strides[7] = (1,1)
conv_filter_nbrs[7] = 256
conv_pooling[7] = False

conv_filter_sizes[8] = (3,3)
conv_filter_strides[8] = (1,1)
conv_filter_nbrs[8] = 256
conv_pooling[8] = True

# =============================================================================
# conv_filter_sizes[9] = (3,3)
# conv_filter_strides[9] = (1,1)
# conv_filter_nbrs[9] = 64
# conv_pooling[9] = True
# 
# conv_filter_sizes[10] = (3,3)
# conv_filter_strides[10] = (1,1)
# conv_filter_nbrs[10] = 128
# conv_pooling[10] = False
# 
# conv_filter_sizes[11] = (3,3)
# conv_filter_strides[11] = (1,1)
# conv_filter_nbrs[11] = 128
# conv_pooling[11] = False
# 
# conv_filter_sizes[12] = (3,3)
# conv_filter_strides[12] = (1,1)
# conv_filter_nbrs[12] = 128
# conv_pooling[12] = False
# 
# conv_filter_sizes[13] = (3,3)
# conv_filter_strides[13] = (1,1)
# conv_filter_nbrs[13] = 256
# conv_pooling[13] = False
# 
# conv_filter_sizes[14] = (3,3)
# conv_filter_strides[14] = (1,1)
# conv_filter_nbrs[14] = 128
# conv_pooling[14] = True
# =============================================================================
# =============================================================================
# 
# conv_filter_sizes[15] = (3,3)
# conv_filter_strides[15] = (1,1)
# conv_filter_nbrs[15] = 256
# conv_pooling[15] = False
# 
# conv_filter_sizes[16] = (3,3)
# conv_filter_strides[16] = (1,1)
# conv_filter_nbrs[16] = 256
# conv_pooling[16] = False
# 
# conv_filter_sizes[17] = (3,3)
# conv_filter_strides[17] = (1,1)
# conv_filter_nbrs[17] = 256
# conv_pooling[17] = True
# =============================================================================


# Fully connected layers
nbr_full_layers = 4
full_nbr_neurons = np.empty(nbr_full_layers, dtype = np.uint16)

full_nbr_neurons[0] = 512
full_nbr_neurons[1] = 512
full_nbr_neurons[2] = 256
full_nbr_neurons[3] = 128


# higher freq number have to be multiples of lower ones.
freq_report_1 = 10 # when calculate training: cost, f1, prec, recall
freq_report_2 = 20 # when plot training f1, prec, recall + training loss
freq_test = 20 # plot training/test loss and test f1, prec, loss
freq_save = 2000 # when save the model

batch_size = 48
batch_size_test = 48

CV_percentage = 0.05

threshold_val = 0.35 # ideal threshold should be max_threshold F1/2 (max over all thresholds)
keep_prob_val = 0.5

grad_method = 'Adam'
grad_val = 1e-3
factor_reduce_grad_val = 2
freq_reduce_grad_val = 2000 # reduce grad step

alpha = 0.1 # parameter in leaky_relu
beta = 0.0005 #0.0005 #regularization parameter for L2
dropout_full_layer = False

use_pixel_mean_normalization = False

use_weights = True 
smoothing_weights_val_init = 0.75 # between 0 and 1. Value of 0 means all weights = 1 (no weights), Value of 1 means orignal weights (inv. prop. to distribution),
smoothing_weights_coeff_multiplier = 0.95
freq_change_smoothing_weight = 500 # update smoothing weights,

# Others param
desired_size_images = (128,128)
#channels_to_consider = ['green']
channels_to_consider = ['red', 'green', 'blue']

########################################################################################################
### Some internal checks, variables and functions that need the hyperparameters
########################################################################################################

reduction_factor_x = int(original_size_images[0] // desired_size_images[0])
reduction_factor_y = int(original_size_images[1] // desired_size_images[1])
if reduction_factor_x < 1 | reduction_factor_y < 1:
	print('The desired image size is larger than the actual input, so we use the whole image')
	desired_size_images = original_size_images
	reduction_factor_x = reduction_factor_y = 1

# Create the smoothing weights vector
label_weights_val = calc_smoothing_weights(y_df, smoothing_weights_val_init, use_weights)
smoothing_weights_val = smoothing_weights_val_init
label_weights_max_value = label_weights_val.max()

# create a callable leaky_relu, so that I can specify the alpha parameter
leaky_relu = lambda input_layer : tf.nn.leaky_relu(input_layer, alpha=alpha)

# number of examples reserved for the the test
how_many_reserved_for_test = int(nbr_training_exs*CV_percentage)

# load the mean of each pixel for normalization
if use_pixel_mean_normalization:
	try:
		colors = ''
		for color in channels_to_consider:
			colors += color[0].upper()
		X_pixels_mean_filename = 'PixelMean_' + labelname[:-4] + '_' + str(desired_size_images[0]) + 'x' + str(desired_size_images[1]) + '_' + colors + '.txt'
		X_pixels_mean_filepath = os.path.join(r'F:\MLdata\HumanProtein', X_pixels_mean_filename)
		X_pixels_mean = np.loadtxt(X_pixels_mean_filepath)
		print(f"Succesfully loaded the Pixel_mean file from {X_pixels_mean_filepath}")
	except:
		print(f"Couldn't find the file for pixel normalization at location {X_pixels_mean_filepath}, will have to do without (could break stuff though, check filename)")
		use_pixel_mean_normalization = False
else:
	X_pixels_mean = np.zeros((desired_size_images[0], desired_size_images[1], len(channels_to_consider)))
	print('No Pixel_mean in use')

########################################################################################################
#### Load images (zip) and labels (csv)
########################################################################################################
# Split into train/test
if use_test:
	if use_cross_validation_from_train_set:
		y_df, y_df_test = split_into_train_test(y_df, how_many_reserved_for_test)
	else:
		y_df_test = pd.read_csv(os.path.join(path_data_FOLDER, labelname_test))
		#batch_size_test = len(y_df_test.index)
		
# Create the place holders for images and labels
with tf.name_scope('x_and_y'):
	x = tf.placeholder(tf.float32, shape = [None, desired_size_images[0], desired_size_images[1], len(channels_to_consider)], name='var_x')
	y_ = tf.placeholder(tf.float32, shape = [None, nbr_classes], name='var_y')
	y_test = tf.placeholder(tf.float32, shape = [None, nbr_classes], name='var_y_test')

########################################################################################################
#### the NN
########################################################################################################

with tf.name_scope('small_training_param'):
	keep_prob = tf.placeholder(tf.float32)
	threshold = tf.placeholder(tf.float32)
	grad_rate = tf.placeholder(tf.float32)
	smoothing_weights = tf.placeholder(tf.float32)
	label_weights = tf.placeholder(tf.float32, shape = [1,28])

with tf.name_scope('conv_layers'):
	conv_filter_nbrs = np.insert(conv_filter_nbrs, 0, len(channels_to_consider))
	
	conv1 = conv_layer(x, 0, conv_filter_strides[0])
	conv2 = conv_layer(conv1, 1, conv_filter_strides[1])
	conv3 = conv_layer(conv2, 2, conv_filter_strides[2])
	conv4 = conv_layer(conv3, 3, conv_filter_strides[3])
	conv5 = conv_layer(conv4, 4, conv_filter_strides[4])
	conv6 = conv_layer(conv5, 5, conv_filter_strides[5])
	conv7 = conv_layer(conv6, 6, conv_filter_strides[6])
	conv8 = conv_layer(conv7, 7, conv_filter_strides[7])
	conv9 = conv_layer(conv8, 8, conv_filter_strides[8])
# =============================================================================
# 	conv10 = conv_layer(conv9, 9, conv_filter_strides[9])
# 	conv11 = conv_layer(conv10, 10, conv_filter_strides[10])
# 	conv12 = conv_layer(conv11, 11, conv_filter_strides[11])
# 	conv13 = conv_layer(conv12, 12, conv_filter_strides[12])
# 	conv14 = conv_layer(conv13, 13, conv_filter_strides[13])
# 	conv15 = conv_layer(conv14, 14, conv_filter_strides[14])
# =============================================================================
# =============================================================================
# 	conv16 = conv_layer(conv15, 15, conv_filter_strides[15])
# 	conv17 = conv_layer(conv16, 16, conv_filter_strides[16])
# 	conv18 = conv_layer(conv17, 17, conv_filter_strides[17])
# =============================================================================

	conv_flat = tf.layers.flatten(conv9, name='conv_layer_flat')

with tf.name_scope('full_layers'):

	full1 = full_layer(conv_flat, 0)
	full2 = full_layer(full1, 1)
	full3 = full_layer(full2, 2)
	full4 = full_layer(full3, 3)

	#y_conv = full_layer(full3, nbr_classes)
	y_conv = tf.layers.dense(inputs = full4, units= nbr_classes, activation = None, \
				kernel_initializer = tf.initializers.truncated_normal(mean = 0, stddev = 0.1), \
				bias_initializer = tf.constant_initializer(0.1), \
				kernel_regularizer = tf.contrib.layers.l2_regularizer(beta), name='full_layer_final')

with tf.name_scope('loss_error'):
	if use_weights:
		#cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = y_conv, multi_class_labels = y_, weights = label_weights))
		cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = y_conv, targets = y_, pos_weight = label_weights))
		#cross_entropy_test = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = y_conv, multi_class_labels = y_test, weights = label_weights))
		cross_entropy_test = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = y_conv, targets = y_test, pos_weight = label_weights))
	else:
		cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = y_conv, multi_class_labels = y_))
		cross_entropy_test = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = y_conv, multi_class_labels = y_test))


# The reg is now included in the TF functions
# =============================================================================
# with tf.name_scope('regularization'):
# 	regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
# 	reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# 	reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
# =============================================================================

with tf.name_scope('training'):
	
	if grad_method == 'Adam':
		train_step = tf.train.AdamOptimizer(grad_rate).minimize(cross_entropy)
	elif grad_method == 'SGD':
		train_step = tf.train.GradientDescentOptimizer(grad_rate).minimize(cross_entropy)
	else:
		train_step = tf.train.GradientDescentOptimizer(grad_rate).minimize(cross_entropy)
	#train_step = tf.train.AdamOptimizer(adamOpt_rate).minimize(cross_entropy + reg_term)

with tf.name_scope('predictions'):
	prediction_score = tf.nn.sigmoid(y_conv)
	#prediction = tf.cond(tf.equal(tf.round(prediction_score - threshold + 0.5), tf.zeros_like(prediction_score[0])), lambda: tf.math.reduce_max(prediction_score), lambda: tf.round(prediction_score - threshold + 0.5)) # predict a 1 if above the threshold, and 0 if it is below
	prediction = tf.round(prediction_score - threshold + 0.5) # predict a 1 if above the threshold, and 0 if it is below
	
	pred_non_zero = tf.count_nonzero(prediction, axis = 1)
	pred_non_zero_mask = tf.equal(pred_non_zero, tf.zeros_like(pred_non_zero))
	pred_0_and_1_max = tf.where(tf.equal(tf.math.reduce_max(prediction_score, axis = 1, keep_dims = True), prediction_score), tf.constant(1, shape = (batch_size, nbr_classes)), tf.constant(0, shape = (batch_size,nbr_classes)))
	
	prediction_corrected = tf.where(pred_non_zero_mask, pred_0_and_1_max, tf.cast(prediction, tf.int32))
	
with tf.name_scope('perf_metrics'):
	TP = tf.count_nonzero(prediction * y_, axis=0)
	TN = tf.count_nonzero((prediction - 1) * (y_ - 1), axis = 0)
	FP = tf.count_nonzero(prediction * (y_ - 1), axis = 0)
	FN = tf.count_nonzero((prediction - 1) * y_, axis = 0)

	TPFPFN_are_zero = tf.logical_and(tf.logical_and(tf.equal(TP,0),tf.equal(FP,0)),tf.equal(FN,0))
	TPFP_are_zero = tf.logical_and(tf.equal(TP,0),tf.equal(FP,0))
	TPFN_are_zero = tf.logical_and(tf.equal(TP,0),tf.equal(FN,0))

	precision_with_nan = TP / (TP + FP)
	precision_with_less_nan = tf.where(TPFP_are_zero, tf.zeros_like(precision_with_nan), precision_with_nan)
	precision = tf.where(TPFPFN_are_zero, tf.ones_like(precision_with_less_nan), precision_with_less_nan)
	precision_reduced = tf.reduce_mean(precision)

	recall_with_nan = TP / (TP + FN)
	recall_with_less_nan = tf.where(TPFN_are_zero, tf.zeros_like(recall_with_nan), recall_with_nan)
	recall = tf.reduce_mean(tf.where(TPFPFN_are_zero, tf.ones_like(recall_with_less_nan), recall_with_less_nan))
	recall_reduced = tf.reduce_mean(recall)

	f1_with_nan = 2*TP / (2*TP + FP + FN)
	macrof1 = tf.reduce_mean(tf.where(TPFPFN_are_zero, tf.ones_like(f1_with_nan), f1_with_nan))

with tf.name_scope('perf_metrics_test'):

	TP_test = tf.count_nonzero(prediction * y_test, axis=0)
	TN_test = tf.count_nonzero((prediction - 1) * (y_test - 1), axis = 0)
	FP_test = tf.count_nonzero(prediction * (y_test - 1), axis = 0)
	FN_test = tf.count_nonzero((prediction - 1) * y_test, axis = 0)

	TPFPFN_are_zero_test = tf.logical_and(tf.logical_and(tf.equal(TP_test,0),tf.equal(FP_test,0)),tf.equal(FN_test,0))
	TPFP_are_zero_test = tf.logical_and(tf.equal(TP_test,0),tf.equal(FP_test,0))
	TPFN_are_zero_test = tf.logical_and(tf.equal(TP_test,0),tf.equal(FN_test,0))

	precision_with_nan_test = TP_test / (TP_test + FP_test)
	precision_with_less_nan_test = tf.where(TPFP_are_zero_test, tf.zeros_like(precision_with_nan_test), precision_with_nan_test)
	precision_test = tf.reduce_mean(tf.where(TPFPFN_are_zero_test, tf.ones_like(precision_with_less_nan_test), precision_with_less_nan_test))

	recall_with_nan_test = TP_test / (TP_test + FN_test)
	recall_with_less_nan_test = tf.where(TPFN_are_zero_test, tf.zeros_like(recall_with_nan_test), recall_with_nan_test)
	recall_test = tf.reduce_mean(tf.where(TPFPFN_are_zero_test, tf.ones_like(recall_with_less_nan_test), recall_with_less_nan_test))

	f1_with_nan_test = 2*TP_test / (2*TP_test + FP_test + FN_test)
	macrof1_test = tf.reduce_mean(tf.where(TPFPFN_are_zero_test, tf.ones_like(f1_with_nan_test), f1_with_nan_test ))

with tf.name_scope('save_the_model'):
	saver = tf.train.Saver(max_to_keep=None)

# =============================================================================
# with tf.name_scope('tensor_board'):
# 	writer = tf.summary.FileWriter('.')
# 	writer.add_graph(tf.get_default_graph())
#
# =============================================================================

########################################################################################################
### The TF session
########################################################################################################



gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#with tf.Session() as sess:

	if load_model_and_predict_for_submission:
		saver.restore(sess, os.path.join(path_load_model, filename_load_model))
		print('System loaded from {}'.format(filename_load_model))
		y_df_pred_test = pd.read_csv(labelname_test)

		batch_size = 1
		nbr_batches = y_df_pred_test['Id'].size

		result = pd.DataFrame(columns=['Id', 'Predicted'])
		
		datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
		datetime_now = datetime_now[:datetime_now.find('.')]
		result_filename = os.path.join(path_save_model, filename_load_model[:-5] + '_result.csv')
		

		for i in range(nbr_batches):
			# if we are at the last batch, we have to be careful as it probably won't be full
			if i == nbr_batches-1:
				pred_test = sess.run(prediction, feed_dict={x: extract_batch_pred_test(y_df_pred_test, zipname_test, desired_size_images, batch_size, i, X_pixels_mean ), threshold:threshold_val})
				local_pred = [idx for idx, val in enumerate(pred_test[0]) if val == 1.0]
				y_df_pred_test['Predicted'][i] = ' '.join(str(e) for e in local_pred)
				y_df_pred_test.to_csv(os.path.join(r'F:\MLdata\HumanProtein', result_filename), index=False )
				print('Prediction done')
				break
			if i % 1000 == 0:
				print('Batch {} / {} done'.format(i, nbr_batches))
				
			pred_test = sess.run(prediction, feed_dict={x: extract_batch_pred_test(y_df_pred_test, zipname_test, desired_size_images, batch_size, i, X_pixels_mean), threshold:threshold_val})
			local_pred = [idx for idx, val in enumerate(pred_test[0]) if val == 1.0]
			y_df_pred_test['Predicted'][i] = ' '.join(str(e) for e in local_pred)
			#y_df_pred_test.to_csv(os.path.join(r'F:\MLdata\HumanProtein', result_filename), index=False )


	else:
		print('Starting to initiate variables')
		sess.run(tf.global_variables_initializer(), options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
		print('Global var initiated')
		sess.run(tf.local_variables_initializer(), options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
		print('Local var initiated')
		print('The settings are: \
		   \n training set with {} elements \
		   \n {} conv layers, {} fully conec layers \
		   \n original size image {} rescaled to {} \
		   \n channels {} \
		   \n batch size of {} \
		   \n threshold at {} \
		   \n Grad method is {}, with value of {} (reduced every {} steps by {})  \
		   \n Dropout: {} (with rate of {}) \
		   \n initial smoothing factor for the weights {}'.format( \
		   nbr_training_exs, \
		   nbr_conv_layers, nbr_full_layers, \
		   original_size_images, desired_size_images, \
		   channels_to_consider, \
		   batch_size, threshold_val , \
		   grad_method, grad_val, freq_reduce_grad_val, factor_reduce_grad_val, \
		   dropout_full_layer, keep_prob_val, smoothing_weights_val_init))

		time_all_loaded = time.time()
		last_time = time_all_loaded
		print(f'All was loaded in {time_all_loaded - time_start} seconds')

		f1_history = []
		prec_history = []
		recall_history = []
		loss_history = []

		f1_history_test = []
		prec_history_test = []
		recall_history_test = []

		loss_history_traintest = [[], []] # test and train

		epoch = 1
		MAX_STEPS = 30000
		
		start_plot_training = 0

		for i in range(1,MAX_STEPS+1):
			#print('Extracting the batch')
			print(i, end=" ", flush=True)
			
			#batch_start_time = time.time()
			batch = extract_batch(y_df, zipname, desired_size_images, batch_size, i-1, X_pixels_mean)
			#batch_end_time = time.time()
			#print(f'Batch extracted in {batch_end_time - batch_start_time} sec')
			feed_dict_train = {x: batch[0], y_: batch[1], grad_rate:grad_val, keep_prob: keep_prob_val, threshold:threshold_val, label_weights: label_weights_val}
			sess.run(train_step, feed_dict=feed_dict_train, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			#print(f'Training step took {time.time() - batch_end_time} sec')

			if i % freq_report_1 == 0:

				calc_prec_time = time.time()

				print_train_cost, print_macrof1, print_prec, print_prec_reduced, print_recall = sess.run([cross_entropy, macrof1, precision, precision_reduced, recall], feed_dict=feed_dict_train)
				
				f1_history.append(print_macrof1)
				prec_history.append(print_prec_reduced)
				recall_history.append(print_recall)
				loss_history.append(print_train_cost)
				
				#print(f'Calculating cost and precision took {time.time() - calc_prec_time} sec')
				
				time_freq_rep1 = time.time()
				print(f'Total time is {time_freq_rep1 - time_all_loaded} sec, {time_freq_rep1 - last_time} for the last {freq_report_1} steps')
				last_time = time_freq_rep1

				if i % freq_report_2 == 0:
					epoch = i * batch_size / nbr_training_exs*(1-CV_percentage) + 1

					plt.figure(figsize=(10,3))
					#plt.ylim(0,0.7)
					plt.plot(f1_history, label = 'f1')
					plt.plot(prec_history, label = 'prec')
					plt.plot(recall_history, label = 'recall')
					plt.legend(loc ='upper left')
					plt.title('train accuracy after {} iterations, {} training examples, epoch {}'.format(i, i*batch_size, epoch))
					plt.show()
					
					if use_test == False:
						plt.figure(figsize=(10,4))
						plt.plot(loss_history, label = 'training cost')
						plt.legend(loc ='upper right')
						plt.title('cost at epoch {}'.format(epoch))
						plt.show()

					if i % freq_change_smoothing_weight == 0:
						if label_weights_val.max() > 1.5:
							smoothing_weights_val *= smoothing_weights_coeff_multiplier
							print('The new smoothing weight is {}'.format(smoothing_weights_val))
							label_weights_val = calc_smoothing_weights(y_df, smoothing_weights_val, use_weights)

				if i % freq_reduce_grad_val == 0:
					if grad_val > 1e-6:
						grad_val = grad_val / factor_reduce_grad_val
						print(f'Reduced the gradient step to {grad_val}')
			
			if i % freq_save == 0:
				epoch = i * batch_size / nbr_training_exs*(1-CV_percentage) + 1
				# customize the name of output, add date&time to avoid overwrite by mistake
				datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
				datetime_now = datetime_now[:datetime_now.find('.')]
				filename_save_model = os.path.join(path_save_model, 'model_' + datetime_now + '.ckpt')
				save_path = saver.save(sess, filename_save_model)
				
				with open(os.path.join(path_save_model, 'model_' + datetime_now + '_details.txt'), 'w') as notepad_file:
					notepad_file.write(f'Model from {datetime.now()}, trained {int((time_freq_rep1 - time_all_loaded) / 60)} min on {i*batch_size} examples ({epoch} epochs).\n')
					notepad_file.write(f'Training set: {nbr_training_exs} elements with channels {channels_to_consider} of original size {original_size_images}, resized to {desired_size_images}.')
					if use_pixel_mean_normalization:
						notepad_file.write(' The images where normalized by subtracting the mean of the whole set, pixel by pixel.')
					notepad_file.write(f'\nTraining: {i} iterations with {grad_method} (rate {grad_val}, reduced every {freq_reduce_grad_val} steps by {factor_reduce_grad_val}), batches of {batch_size} images and L2 regularization with parameter {beta}.\n')
					if dropout_full_layer:
						notepad_file.write(f'Dropout: we used dropout in the fully connected layers with rate {keep_prob_val}.\n')
					if use_weights:
						notepad_file.write(f'Weights: the smoothing weight coeff started at {smoothing_weights_val_init} with max weight of {label_weights_max_value}. Every {freq_change_smoothing_weight} iterations it is multiplied by {smoothing_weights_coeff_multiplier}. It\'s value is now {smoothing_weights_val} and the max weight has value {label_weights_val.max()}.\n')
					notepad_file.write(f'Architecture: {nbr_conv_layers} conv layers followed by {nbr_full_layers} fully connected layers, all have leaky ReLU activation with parameter {alpha}.\n')
					notepad_file.write('More precisely and in this order the hidden layers are:')
					for j in range(nbr_conv_layers):
						notepad_file.write(f'\n - conv layer {j+1} with {conv_filter_nbrs[j+1]} filters of size {conv_filter_sizes[j]} with stride {conv_filter_strides[j]}')
						if conv_pooling[j]:
							notepad_file.write(' with 2x2 max pooling')
					for j in range(nbr_full_layers):
						notepad_file.write(f'\n - fully connected layer {j+1} with {full_nbr_neurons[j]} neurons')
					notepad_file.write('\n\nSize: ')
					notepad_file.write('\n\nSUBMIT SCORE: ')

				print("Model saved in path: %s" % save_path)

			if i % freq_test == 0:
				if use_test:
					batch_test = extract_batch(y_df_test, zipname_test, desired_size_images, batch_size_test, int(i // freq_test) -1, X_pixels_mean) # ADD RANDOM
					print_test_cost, print_macrof1_test, print_prec_test, print_recall_test = sess.run([cross_entropy_test, macrof1_test, precision_test, recall_test], feed_dict={x: batch_test[0], y_test: batch_test[1], keep_prob: 1.0, threshold: threshold_val, label_weights: label_weights_val})
					
					f1_history_test.append(print_macrof1_test)
					prec_history_test.append(print_prec_test)
					recall_history_test.append(print_recall_test )
					
					ratio_calc_plot = int(freq_test // freq_report_1)
					
					x_axis_range = list(range(0, len(f1_history_test) * ratio_calc_plot, ratio_calc_plot))
					plt.figure(figsize=(10,3))
					plt.plot(x_axis_range, f1_history_test, label = 'f1')
					plt.plot(x_axis_range, prec_history_test, label = 'prec')
					plt.plot(x_axis_range, recall_history_test, label = 'recall')
					plt.legend(loc ='upper left')
					plt.title('test accuracy at epoch {}'.format(epoch))
					plt.show()
					#print('Test cost at step {} is {}'.format(i, print_test_cost))
					
					loss_history_traintest[0].append(print_train_cost)
					loss_history_traintest[1].append(print_test_cost)
					if (loss_history_traintest[0][start_plot_training] - loss_history_traintest[0][-1]) / loss_history_traintest[0][start_plot_training] > 0.5:
						start_plot_training = int((len(loss_history_traintest[0]) + start_plot_training) / 2)
					
					x_axis_new_range = list(range(start_plot_training * ratio_calc_plot, len(loss_history_traintest[0]) * ratio_calc_plot, ratio_calc_plot))
					plt.figure(figsize=(10,4))
					plt.plot(x_axis_new_range, loss_history_traintest[0][start_plot_training:], label = 'training cost')
					plt.plot(x_axis_new_range, loss_history_traintest[1][start_plot_training:], label = 'test cost')
					plt.legend(loc ='upper right')
					plt.title('cost at epoch {} (zoomed in last epochs)'.format(epoch))
					plt.show()

