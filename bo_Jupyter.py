# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:57:43 2019

@author: gheor
"""

# Import some modules
import pandas as pd
import numpy as np
import os
from datetime import datetime
from PIL import Image
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2

np.set_printoptions(precision=0, suppress=True, threshold=np.nan)

# Some Global Variables
path_data_FOLDER = r'C:\MLdata\HAP'
path_train_data = os.path.join(path_data_FOLDER, 'train')
path_labels = os.path.join(path_data_FOLDER, 'train.csv')

channels_to_consider = ['red', 'green', 'blue', 'yellow']
grad_methods = ['SGD', 'Adam', 'Adagrad']
original_size_images = (512,512)
nbr_classes = 28

organelle_dict = {0 :  'Nucleoplasm',
    1:  'Nuclear membrane',
    2:  'Nucleoli',
    3:  'Nucleoli fibrillar center',
    4:  'Nuclear speckles',
    5:  'Nuclear bodies',
    6:  'Endoplasmic reticulum',
    7:  'Golgi apparatus',
    8:  'Peroxisomes',
    9:  'Endosomes',
    10:  'Lysosomes',
    11:  'Intermediate filaments',
    12:  'Actin filaments',
    13:  'Focal adhesion sites',
    14:  'Microtubules',
    15:  'Microtubule ends',
    16:  'Cytokinetic bridge',
    17:  'Mitotic spindle',
    18:  'Microtubule organizing center',
    19:  'Centrosome',
    20:  'Lipid droplets',
    21:  'Plasma membrane',
    22:  'Cell junctions',
    23:  'Mitochondria',
    24:  'Aggresome',
    25:  'Cytosol',
    26:  'Cytoplasmic bodies',
    27:  'Rods & rings'}

###########################################################################

def plot_distribution(path_labels_to_plot = path_labels):
	
	y_df = pd.read_csv(path_labels_to_plot)
	# change in one-hot encoding
	for i in range(28):
	    y_df[f'{i}'] = y_df['Target'].map(lambda x: 1 if str(i) in x.strip().split() else 0)
	
	target_counts = y_df.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
	# Distribution of labels
	plt.figure(figsize=(10,8))
	plt.title('Graphical distribution of the labels:')
	sns.barplot(x=target_counts.index.values, y=target_counts.values) #, order=target_counts.index)
	plt.show()
	print(f'Here is the distribution of labels: \n {target_counts}')
	
	nbr_classes_per_image = y_df.drop(["Id", "Target"], axis=1).sum(axis=1)
	
	print('Graphical distribution of the number of labels per image:')
	plt.xticks(np.unique(nbr_classes_per_image))
	plt.hist(nbr_classes_per_image, bins = np.unique(nbr_classes_per_image), label = 'Graphical distribution of the number of labels per image:')
	plt.show()


def extract_balanced_sample_with_test(nbr_files = None, ratio_files_test = 0.2):
	
	y_df = pd.read_csv(path_labels)
	y_df['ReflectRotate'] = 0
	nbr_training_exs = y_df['Id'].size
	
	if nbr_files == None:
		nbr_files = nbr_training_exs
	if ratio_files_test > 0.5:
		print('The ratio for the test set is too high, set it to 0.5')
		ratio_files_test = 0.5
	
	nbr_files_test = int(ratio_files_test * nbr_files)
	nbr_files_train = nbr_files - nbr_files_test

	nbr_images_per_label_train = math.ceil(nbr_files_train // nbr_classes)
	nbr_images_per_label_test = math.ceil(nbr_files_test // nbr_classes)
	nbr_images_per_label = nbr_images_per_label_train + nbr_images_per_label_test
	
	rdm = 1 # the random state for picking sample, could put a random number...
	
	y_df_train = pd.DataFrame(columns = y_df.columns)
	y_df_test = pd.DataFrame(columns = y_df.columns)
	
	for i in range(nbr_classes):
		y_df[f'{i}'] = y_df['Target'].map(lambda x: 1 if str(i) in x.strip().split() else 0)
		if len(y_df[y_df[str(i)] == 1].index) >= nbr_images_per_label:
			print(f'For label {i} there are enough: { len(y_df[y_df[str(i)] == 1].index)} bigger than {nbr_images_per_label}')
			y_df_train = pd.concat([y_df_train, y_df[y_df[str(i)] == 1][['Id', 'Target', 'ReflectRotate']].sample(n=nbr_images_per_label, random_state = rdm).reset_index(drop=True)[:nbr_images_per_label_train]]).reset_index(drop=True)
			y_df_test = pd.concat([y_df_test, y_df[y_df[str(i)] == 1][['Id', 'Target', 'ReflectRotate']].sample(n=nbr_images_per_label, random_state = rdm).reset_index(drop=True)[nbr_images_per_label_train:nbr_files]]).reset_index(drop=True)

		else:
			print(f'For label {i} there are not enough: { len(y_df[y_df[str(i)] == 1].index)} less than {nbr_images_per_label}')
			
			nbr_images_from_df_for_train = math.ceil(len(y_df[y_df[str(i)] == 1].index) * (1 - ratio_files_test))
			nbr_images_from_df_for_test = len(y_df[y_df[str(i)] == 1].index) - nbr_images_from_df_for_train
			
			if nbr_images_from_df_for_test == 0:
				nbr_images_from_df_for_train = nbr_images_from_df_for_train -1
				nbr_images_from_df_for_test = 1
			
			for j in range(math.ceil(nbr_images_per_label_train / nbr_images_from_df_for_train)):
				y_df_local = y_df[y_df[str(i)] == 1][:nbr_images_from_df_for_train].copy(deep=True)
				y_df_local['ReflectRotate'] = j % 7 # As there are 7 transformation
				y_df_train = pd.concat([y_df_train, y_df_local[['Id', 'Target', 'ReflectRotate']].reset_index(drop=True)]).reset_index(drop=True)

			for j in range(math.ceil(nbr_images_per_label_test / nbr_images_from_df_for_test)):
				y_df_local = y_df[y_df[str(i)] == 1][nbr_images_from_df_for_train:].copy(deep=True)
				y_df_local['ReflectRotate'] = j % 7 # As there are 7 transformation
				y_df_test = pd.concat([y_df_test, y_df_local[['Id', 'Target', 'ReflectRotate']].reset_index(drop=True)]).reset_index(drop=True)
			
			print(f'For label {i}: we had to multiply the data (adding reflections/rotations) by {int(nbr_images_per_label // len(y_df[y_df[str(i)] == 1].index)) +1}')

	# use unique date&time for the name of the created .csv
	datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
	datetime_now = datetime_now[:datetime_now.find('.')]
	filename_train =  'train_' + datetime_now + '_' + str(nbr_files_train) + 'files_balanced.csv'
	filename_test =  'train_' + datetime_now + '_' + str(nbr_files_test) + 'files_balanced_test_set.csv'

	y_df_train.sample(n=nbr_files_train, replace=True).reset_index(drop=True)
	y_df_train.to_csv(os.path.join(path_data_FOLDER, filename_train), index = False)
	
	y_df_test.sample(n=nbr_files_test, replace=True).reset_index(drop=True)
	y_df_test.to_csv(os.path.join(path_data_FOLDER, filename_test), index = False)

	print(f'Succesfully saved the file of {nbr_files} images in {path_data_FOLDER}')


plot_distribution()
extract_balanced_sample_with_test()
plot_distribution(os.path.join(path_data_FOLDER, 'train_20190102143833_24858files_balanced.csv'))
