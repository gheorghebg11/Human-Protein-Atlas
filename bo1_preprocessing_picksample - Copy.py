# -*- coding: utf-8 -*-
"""
Bo:
	- pick a small sample
	- calculate mean for each pixel in image for normalization
"""


#Load libraries
import zipfile as zf
import pandas as pd
import numpy as np
import os
from datetime import datetime
import tempfile
from PIL import Image
import math

np.set_printoptions(threshold=np.nan)

#####################################################
## Some prerequisites
#####################################################

onDesktop = True
onDesktop_onSSD = True
unzipped = True

originalSet = True
extended = False

### Set The folders
original_size_images = (512,512)

#path_data_FOLDER = r'C:\Users\Bogdan\Desktop\dataProtein'
if onDesktop:
	path_data_FOLDER = r'F:\MLdata\HumanProtein'
	if onDesktop_onSSD and unzipped:
		path_data_FOLDER = r'C:\MLdata'
else:
	path_data_FOLDER = os.path.join(os.getcwd(), 'smalldata')
	if originalSet:
		path_data_FOLDER = r'C:\Users\Bogdan\Desktop\dataProtein'

if originalSet:
	file = 'train'
	original_size_images = (512,512)

### Set The Channels & Nbr of Labels
channels = ['yellow', 'red', 'green', 'blue']
nbr_classes = 28

if unzipped:
	filename = 'train'
else:
	### Loading all of this
	zipname = file + '.zip'

labelname = file + '.csv'

if extended:
	labelname = 'train_labels_extended20181210171542.csv'
	
y_df = pd.read_csv(os.path.join(path_data_FOLDER, labelname))
nbr_training_exs = y_df['Id'].size

#y_df = y_df.sample(frac=1).reset_index(drop=True) # just shuffling the dataset



path_data_FOLDER_read = r'C:\Users\Bogdan\Desktop\dataProtein'
path_data_FOLDER_write = path_data_FOLDER_read
if onDesktop:
	path_data_FOLDER_read = r'F:\MLdata\HumanProtein'
	#path_data_FOLDER_write = r'F:\Dropbox\Machine Learning\ML projects\Kaggle - Human Protein Atlas Image Classification'
	path_data_FOLDER_write = path_data_FOLDER_read
	

# Setting up the folders path
path_train_data = os.path.join(path_data_FOLDER, filename)

#####################################################
## Create a zip file with less images 
#####################################################

def extract_balanced_sample(nbr_files, size, path_to_file):
	
	# customize the name of output, add date&time to avoid overwrite by mistake
	datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
	datetime_now = datetime_now[:datetime_now.find('.')]
	#print(datetime_now)
	smaller_file_name = filename +  '_' + str(nbr_files) + 'files_' + '_' + datetime_now + '.zip'
	path_train_data_smaller = os.path.join(path_data_FOLDER_write, smaller_file_name)
	
	nbr_images_per_label = int(nbr_files // nbr_classes) + 1
	
	path_train_labels = os.path.join(path_data_FOLDER, labelname)
	
	y_df = pd.read_csv(path_train_labels)
	y_df['ReflectRotate'] = 0
	y_df_smaller = pd.DataFrame(columns = y_df.columns)
	

	#with zf.ZipFile(path_train_data_smaller, 'w') as zip_train_smaller:
	
	for i in range(nbr_classes):
		y_df[f'{i}'] = y_df['Target'].map(lambda x: 1 if str(i) in x.strip().split() else 0)
		if len(y_df[y_df[str(i)] == 1].index) >= nbr_images_per_label:
			print(f'For label {i} there are enough: { len(y_df[y_df[str(i)] == 1].index)} bigger than {nbr_images_per_label}')
			y_df_smaller = pd.concat([y_df_smaller, y_df[y_df[str(i)] == 1][['Id', 'Target', 'ReflectRotate']].sample(n=nbr_images_per_label).reset_index(drop=True)]).reset_index(drop=True)
		else:
			print(f'For label {i} there are not enough: { len(y_df[y_df[str(i)] == 1].index)} less than {nbr_images_per_label}')
			for j in range(int(nbr_images_per_label // len(y_df[y_df[str(i)] == 1].index)) +1):
				y_df_local = y_df[y_df[str(i)] == 1].copy(deep=True)
				y_df_local['ReflectRotate'] = j % 7 # As there are 7 transformation
				y_df_smaller = pd.concat([y_df_smaller, y_df_local[['Id', 'Target', 'ReflectRotate']].reset_index(drop=True)]).reset_index(drop=True)

			print(f'Label {i} had to be multiplied by {int(nbr_images_per_label // len(y_df[y_df[str(i)] == 1].index)) +1}')

	colors = ['red', 'green', 'blue', 'yellow']
			
			
# =============================================================================
# 		with tempfile.TemporaryDirectory() as dir:
# 			os.chdir(dir)
# 	
# 			for index, file in y_df_smaller['Id'].iteritems():
# 				print('{} / {} {}'.format(index + 1, nbr_files, file))
# 				for color in colors:
# 					temp_filename = os.path.join(path_train_data, file + '_' + color + '.png')
# 
# 					with Image.open(temp_filename) as original_image:
# 						original_image.thumbnail(size, Image.ANTIALIAS)
# 						original_image.save(temp_filename)
# 
# 					zip_train_smaller.write(temp_filename)
# 			os.chdir(path_data_FOLDER_write)
# =============================================================================
	y_df_smaller = y_df_smaller.sample(n=nbr_files).reset_index(drop=True)
	y_df_smaller.to_csv(filename +  '_' + str(nbr_files) + 'files_balanced_' + datetime_now + '.csv')

		
# =============================================================================
# 	if unzipped == False: ### DEPRECATED, ONLY WORKS IN UNZIPPED MDOE
# 		with zf.ZipFile(path_to_file, 'r') as zip_train, zf.ZipFile(path_train_data_smaller, 'w') as zip_train_smaller:
# 		
# 			#paths_files = zip_train.namelist()[:nbr_files]
# 			path_train_labels = os.path.join(path_data_FOLDER_read, labelname)
# 			y_df_smaller = pd.read_csv(path_train_labels).sample(n=nbr_files).sort_index().reset_index(drop=True)
# 		
# 			colors = ['red', 'green', 'blue', 'yellow']
# 		
# 			with tempfile.TemporaryDirectory() as dir:
# 				os.chdir(dir)
# 				for index, file in y_df_smaller['Id'].iteritems():
# 					print('{} / {} {}'.format(index + 1, nbr_files, file))
# 					for color in colors:
# 						temp_filename = file + '_' + color + '.png'
# 		
# 						zip_train.extract(temp_filename , path = dir)
# 		
# 						with Image.open(temp_filename) as original_image:
# 							original_image.thumbnail(size, Image.ANTIALIAS)
# 							original_image.save(temp_filename)
# 		
# 						zip_train_smaller.write(temp_filename)
# 				os.chdir(path_data_FOLDER_write)
# 			y_df_smaller.to_csv(filename[:-4] +  '_' + str(nbr_files) + 'files_' + str(size[0]) +'x' + str(size[1]) + '_' + datetime_now + '.csv', index = False)
# 	
# =============================================================================
	print(f'Succesfully saved the file of {nbr_files} images at {path_train_data_smaller}')
	
def extract_balanced_sample_with_test(nbr_files_train, nbr_files_test):
	
	# customize the name of output, add date&time to avoid overwrite by mistake
	datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
	datetime_now = datetime_now[:datetime_now.find('.')]
	#print(datetime_now)
	filename_train =  'train_' + datetime_now + '_' + str(nbr_files_train) + 'files_balanced.csv'
	filename_test =  'train_' + datetime_now + '_' + str(nbr_files_test) + 'files_balanced_disjoint_test_set.csv'
	
	nbr_files = nbr_files_train + nbr_files_test
	ratio_test_in_traintest = nbr_files_test / (nbr_files_test + nbr_files_train)
	
	nbr_images_per_label_train = math.ceil(nbr_files_train // nbr_classes)
	nbr_images_per_label_test = math.ceil(nbr_files_test // nbr_classes)
	nbr_images_per_label = nbr_images_per_label_train + nbr_images_per_label_test
	
	path_train_labels = os.path.join(path_data_FOLDER, labelname)
	
	y_df = pd.read_csv(path_train_labels)
	y_df['ReflectRotate'] = 0
	y_df_train = pd.DataFrame(columns = y_df.columns)
	y_df_test = pd.DataFrame(columns = y_df.columns)
	
	rdm = 1 # the random state for picking sample, could put a random number...

	#with zf.ZipFile(path_train_data_smaller, 'w') as zip_train_smaller:
	
	for i in range(nbr_classes):
		y_df[f'{i}'] = y_df['Target'].map(lambda x: 1 if str(i) in x.strip().split() else 0)
		if len(y_df[y_df[str(i)] == 1].index) >= nbr_images_per_label:
			print(f'For label {i} there are enough: { len(y_df[y_df[str(i)] == 1].index)} bigger than {nbr_images_per_label}')
			y_df_train = pd.concat([y_df_train, y_df[y_df[str(i)] == 1][['Id', 'Target', 'ReflectRotate']].sample(n=nbr_images_per_label, random_state = rdm).reset_index(drop=True)[:nbr_images_per_label_train]]).reset_index(drop=True)
			y_df_test = pd.concat([y_df_test, y_df[y_df[str(i)] == 1][['Id', 'Target', 'ReflectRotate']].sample(n=nbr_images_per_label, random_state = rdm).reset_index(drop=True)[nbr_images_per_label_train:nbr_files]]).reset_index(drop=True)

		else:
			print(f'For label {i} there are not enough: { len(y_df[y_df[str(i)] == 1].index)} less than {nbr_images_per_label}')
			
			nbr_images_from_df_for_train = math.ceil(len(y_df[y_df[str(i)] == 1].index) * (1 - ratio_test_in_traintest))
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
			
			print(f'Label {i} had to be multiplied by {int(nbr_images_per_label // len(y_df[y_df[str(i)] == 1].index)) +1}')


			

	y_df_train.sample(n=nbr_files_train, replace=True).reset_index(drop=True)
	y_df_train.to_csv(os.path.join(path_data_FOLDER, filename_train), index = False)
	
	y_df_test.sample(n=nbr_files_test, replace=True).reset_index(drop=True)
	y_df_test.to_csv(os.path.join(path_data_FOLDER, filename_test), index = False)

	print(f'Succesfully saved the file of {nbr_files} images in {path_data_FOLDER}')




def extract_sample(nbr_files, size, path_to_file): 
	
	# customize the name of output, add date&time to avoid overwrite by mistake
	datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
	datetime_now = datetime_now[:datetime_now.find('.')]
	#print(datetime_now)
	smaller_file_name = filename[:-4] +  '_' + str(nbr_files) + 'files_' + str(size[0]) +'x' + str(size[1]) + '_' + datetime_now + '.zip'
	path_train_data_smaller = os.path.join(path_data_FOLDER_write, smaller_file_name)

	if unzipped:
		with zf.ZipFile(path_train_data_smaller, 'w') as zip_train_smaller:
			path_train_labels = os.path.join(path_data_FOLDER, labelname)
			y_df_smaller = pd.read_csv(path_train_labels).sample(n=nbr_files).sort_index().reset_index(drop=True)
			
			colors = ['red', 'green', 'blue', 'yellow']
			
			with tempfile.TemporaryDirectory() as dir:
				os.chdir(dir)

				for index, file in y_df_smaller['Id'].iteritems():
					print('{} / {} {}'.format(index + 1, nbr_files, file))
					for color in colors:
						temp_filename = os.path.join(path_train_data, file + '_' + color + '.png')
		
						with Image.open(temp_filename) as original_image:
							original_image.thumbnail(size, Image.ANTIALIAS)
							original_image.save(temp_filename)
		
						zip_train_smaller.write(temp_filename)
				os.chdir(path_data_FOLDER_write)
			y_df_smaller.to_csv(filename +  '_' + str(nbr_files) + 'files_' + str(size[0]) +'x' + str(size[1]) + '_' + datetime_now + '.csv', index = False)

		
	else:
		with zf.ZipFile(path_to_file, 'r') as zip_train, zf.ZipFile(path_train_data_smaller, 'w') as zip_train_smaller:
		
			#paths_files = zip_train.namelist()[:nbr_files]
			path_train_labels = os.path.join(path_data_FOLDER_read, labelname)
			y_df_smaller = pd.read_csv(path_train_labels).sample(n=nbr_files).sort_index().reset_index(drop=True)
		
			colors = ['red', 'green', 'blue', 'yellow']
		
			with tempfile.TemporaryDirectory() as dir:
				os.chdir(dir)
				for index, file in y_df_smaller['Id'].iteritems():
					print('{} / {} {}'.format(index + 1, nbr_files, file))
					for color in colors:
						temp_filename = file + '_' + color + '.png'
		
						zip_train.extract(temp_filename , path = dir)
		
						with Image.open(temp_filename) as original_image:
							original_image.thumbnail(size, Image.ANTIALIAS)
							original_image.save(temp_filename)
		
						zip_train_smaller.write(temp_filename)
				os.chdir(path_data_FOLDER_write)
			y_df_smaller.to_csv(filename[:-4] +  '_' + str(nbr_files) + 'files_' + str(size[0]) +'x' + str(size[1]) + '_' + datetime_now + '.csv', index = False)
	
	print(f'Succesfully saved the file of {nbr_files} images at {path_train_data_smaller}')


#####################################################
## Compute the mean per pixel
#####################################################

def load_image(filepath, new_size_images, reflecrotate):

	image = Image.open(filepath).resize((new_size_images[0], new_size_images[1]), Image.ANTIALIAS)

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

def compute_mean_per_pixel(Y_df, path_to_images, new_size_images, channels_to_consider):
	global path_data_FOLDER_write, labelname
	
	if 'ReflectRotate' not in Y_df.columns:
		Y_df['ReflectRotate'] = 0
		print('Addded reflec rotate column')
		
	nbr_images = len(Y_df['Id'])
	pixels_sum = np.zeros((len(channels_to_consider), new_size_images[0], new_size_images[1]), dtype=np.uint32)
	
	print(f'Starting to calculate the mean per pixel for {nbr_images} images in size {new_size_images} with channels {channels_to_consider}.')
	print(f"img 0 / {nbr_images} done")
	
	for j in range(nbr_images):
		for i in range(len(channels_to_consider)):
			img_as_array = np.array(load_image(os.path.join(path_to_images, 'train', Y_df['Id'][j] + '_' + channels_to_consider[i] + '.png'), new_size_images, Y_df['ReflectRotate'][j] ))
			pixels_sum[i] += img_as_array
			#pixels_sum += pixels_sum.transpose(2,0,1)[i] =
			#print(Y_df['Id'][j] + '_' + channels_to_consider[i] + '.png', img_as_array)
			#np.savetxt(os.path.join(path_data_FOLDER_write, Y_df['Id'][j] + '_' + channels_to_consider[i] + '.txt'), img_as_array, fmt = '%d')
		if j % 1000 == 0 and j > 0:
			print(f"img {j} / {nbr_images} done")
	
	pixels_avg = np.zeros((len(channels_to_consider), new_size_images[0], new_size_images[1]), dtype=np.float64)
	pixels_avg = (pixels_sum / nbr_images).reshape((new_size_images[0]* new_size_images[1], len(channels_to_consider)))

	
	
	colors = ''
	for color in channels_to_consider:
		colors += color[0].upper()
	
	filename_save = 'PixelMean_' + labelname[:-4] + '_' + str(new_size_images[0]) + 'x' + str(new_size_images[1]) + '_' + colors + '.txt'
	path_save = os.path.join(path_data_FOLDER_write, filename_save)
	
	# Precision for saving it. Adding 1 or 2 decimals doubles the size of the file...
	np.savetxt(path_save, pixels_avg, fmt = '%.2f')
	
	print(f'Computed mean per pixel for {nbr_images} images in size {new_size_images} with channels {channels_to_consider}. Saved in global {path_data_FOLDER_write}')
	
	
	
	
	
#####################################################
## Set some parameters
#####################################################

# Parameters to set
nbr_files_to_extract = 15000

nbr_files_to_extract_train = 30000
nbr_files_to_extract_test =  nbr_training_exs- nbr_files_to_extract_train

new_size = (512,512)
channels_to_consider = ['red', 'green', 'blue']
channels_to_consider = ['red', 'green', 'blue', 'yellow']


#####################################################
## Call the functions
#####################################################


extract_balanced_sample_with_test(nbr_files_to_extract_train, nbr_files_to_extract_test)

print('DOOOOOOOOOOOOOOONE')


extract_balanced_sample(nbr_files_to_extract, new_size, path_train_data)



for new_size in [(128,128), (256, 256)]:
	new_size = (512,512)
	for channels_to_consider in [['green'], ['red', 'green', 'blue', 'yellow'], ['red', 'green', 'blue']]:
		channels_to_consider = ['green']
		compute_mean_per_pixel(y_df, path_data_FOLDER, new_size, channels_to_consider)
		break
	break



