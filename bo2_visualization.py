# -*- coding: utf-8 -*-
"""
Bo: file adapted from nico_visualization_patches-v2.py. Thus contains:
    -preprocessing
    -visualization
    -a simple CNN (adapted from your file Bo') to classify the data)
    -improved to consider patches of the original images, for efficiency
"""


#Load libraries
import zipfile as zf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
from PIL import Image
import seaborn as sns
import math
from datetime import datetime


###############################################################
### Setting the param to Load the data
###############################################################

#path_data_FOLDER = r'C:\Users\Bogdan\Desktop\dataProtein'
path_data_FOLDER = os.path.join(os.getcwd(), 'smalldata')
filename = 'train_20181223111505_30000files_balanced'

onDesktop = True
onDesktop_onSSD = True

if onDesktop:
	path_data_FOLDER = r'F:\MLdata\HumanProtein'
	if onDesktop_onSSD and unzipped:
		path_data_FOLDER = r'C:\MLdata'
else:
	path_data_FOLDER = os.path.join(os.getcwd(), 'smalldata')


size_images = (512,512)

zipname = filename + '.zip'
labelname = filename + '.csv'

channels = ['_yellow', '_red', '_green', '_blue']

# Create a dictionnary between numbers and biological entities
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
	

###############################################################
### Some functions to plot the data
###############################################################

# Transform the target (string of numbers) into a string of names, for ploting purposes only
def multiple_organelle(string):
    list_orga = string.split(' ')
    organelles = organelle_dict[int(list_orga[0])]
    for num in list_orga[1:]:
        organelles += ' + ' + organelle_dict[int(num)]
    return organelles

def plot_from_zip(arXiv, Y_df, name):
	
    try:
        nrow = len(name)
        plt.figure(figsize=(10,3*nrow))
        for i in range(nrow):
            plt.subplot(nrow, 4, 4*i+1)
            imgfile = arXiv.open(name[i] + '_red.png')
            img = mpimg.imread(imgfile)
            imgplot = plt.imshow(img)
            plt.title(name[i], fontsize = 10)

            plt.subplot(nrow, 4, 4*i+2)
            imgfile = arXiv.open(name[i]+ '_green.png')
            img = mpimg.imread(imgfile)
            imgplot = plt.imshow(img)
            plt.title('label = ' + Y_df[Y_df['Id'] == name[i]].Target.values[0], fontsize = 10)

            plt.subplot(nrow, 4, 4*i+3)
            imgfile = arXiv.open(name[i]+ '_blue.png')
            img = mpimg.imread(imgfile)
            imgplot = plt.imshow(img)
            plt.title(multiple_organelle(Y_df[Y_df['Id'] == name[i]].Target.values[0]), fontsize = 10)

            plt.subplot(nrow, 4, 4*i+4)
            imgfile = arXiv.open(name[i]+ '_yellow.png')
            img = mpimg.imread(imgfile)
            imgplot = plt.imshow(img)
            plt.title('R-G-B-Y', fontsize = 10)
			
        print(f'\nVisualisation of {nrow} images with some of their info:')
        plt.show()
			
    except:
        print('Problem in plot')
        return imgplot
	
def make_rgb_image_from_four_channels(channelsYRGB: list, image_width=512, image_height=512) -> np.ndarray:
    """
    It makes literally RGB image from source four channels, 
    where yellow image will be yellow color, red will be red and so on  
    """
    rgb_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.float)
    yellow = np.array(Image.open(channelsYRGB[0]))
    # yellow is red + green
    rgb_image[:, :, 0] += yellow/2   
    rgb_image[:, :, 1] += yellow/2
    # loop for R,G and B channels
    for index, channel in enumerate(channelsYRGB[1:]):
        current_image = Image.open(channel)
        rgb_image[:, :, index] += current_image
    # Normalize image
    rgb_image = rgb_image / rgb_image.max() * 255
    return rgb_image.astype(np.uint8)


def visualize_part(Y_df, start_class_index=0, nbr_cat=4, nbr_samples_per_cat=3):
    """
    Visualize the part of classes, started from class with index start_class_index,
    make nrows classes, ncols examples for each one
    """
    fig, ax = plt.subplots(nrows = nbr_cat, ncols=nbr_samples_per_cat, figsize=(15, 15))
    for class_index in range(nbr_cat):
        current_index = class_index + start_class_index
		
        for sample in range(nbr_samples_per_cat):
            current_part = Y_df[Y_df[organelle_dict[current_index]] == 1] 
            # 0 index is id
            random_index = np.random.choice(current_part.values.shape[0], 1, replace=False)
            # random line from data with selected class
            current_line = current_part.values[random_index][0]
            
            image_names = [os.path.join(path_data_FOLDER, filename, current_line[0]) + x + '.png' for x in channels]
            
            rgb_image = make_rgb_image_from_four_channels(image_names, size_images[0], size_images[1])
            # text annotations, main title and subclasses (may be empty in case one label)
            main_class = organelle_dict[current_index]+'\n'
            # 2 index is vector with classes, split version of Target col
            other_classes = [organelle_dict[x] for x in current_line[2] 
                             if x != (current_index)]
            subtitle = ', '.join(other_classes)
            # show image
            ax[class_index, sample].set_title(main_class, fontsize=12)
            ax[class_index, sample].text(150, -10, subtitle, 
                                         fontsize=10, horizontalalignment='center')
            ax[class_index, sample].imshow(rgb_image)
            ax[class_index, sample].set_xticklabels([])
            ax[class_index, sample].set_yticklabels([])
            ax[class_index, sample].tick_params(left=False, bottom=False)

def expand_data_reflections(Y_df, threshold_for_rare, max_multiplier, min_even_if_contain_0):
	target_counts = Y_df.drop(["Id", "Target", "ReflectRotate"], axis=1).sum(axis=0).sort_values(ascending=False)
	average_count = target_counts.mean()
	print(f'The average number of examples per label is {average_count}, threshold is {threshold_for_rare * average_count}')
	mask_rare_labels = target_counts[target_counts < threshold_for_rare * average_count]
	
	list_of_df_with_rare_labels = [Y_df]
	
	for rare_label in mask_rare_labels.index.tolist():
		df_rare_label = Y_df[Y_df[str(rare_label)] == 1]
		
		nbr_copies_each_flip = int(min(max_multiplier, threshold_for_rare * average_count / len(df_rare_label.index)))
		#nbr_copies_each_flip = math.ceil(min(max_multiplier, average_count / len(df_rare_label.index)) / 3)
		
		print(f'There are {len(df_rare_label.index)} rare examples, for the label {rare_label}')
		if len(df_rare_label.index) > min_even_if_contain_0:
			df_rare_label = df_rare_label[df_rare_label['0'] == 0]
			df_rare_label = df_rare_label[df_rare_label['25'] == 0]
		print(f'There are {len(df_rare_label.index)} rare examples after removing the ones containsing 0 and 25')
		
		df_rare_label_noflip = df_rare_label.copy(deep = True)
		list_of_df_with_rare_labels.extend([df_rare_label_noflip] * (nbr_copies_each_flip-1))
		
		df_rare_label_flippedH = df_rare_label.copy(deep = True)
		df_rare_label_flippedH.loc[:,'ReflectRotate'] = 1
		list_of_df_with_rare_labels.extend([df_rare_label_flippedH] * nbr_copies_each_flip)
		
		df_rare_label_flippedV = df_rare_label.copy(deep = True)
		df_rare_label_flippedV.loc[:,'ReflectRotate'] = 2
		list_of_df_with_rare_labels.extend([df_rare_label_flippedV] * nbr_copies_each_flip)		
		
		df_rare_label_flippedHV = df_rare_label.copy(deep = True)
		df_rare_label_flippedHV.loc[:,'ReflectRotate'] = 3
		list_of_df_with_rare_labels.extend([df_rare_label_flippedHV] * nbr_copies_each_flip)
		
		print(f'Label {rare_label} has only {len(df_rare_label.index)} examples, we multiply it by {4*nbr_copies_each_flip} by incorporating: H flips, V flips and H+V flips')

	extended_df = pd.concat(list_of_df_with_rare_labels).reset_index(drop=True)
	print(f'There are now {len(extended_df.index)} examples')
	return extended_df

def expand_data_rotations(Y_df, threshold_for_rare, max_multiplier, min_even_if_contain_0):
	target_counts = Y_df.drop(["Id", "Target", "ReflectRotate"], axis=1).sum(axis=0).sort_values(ascending=False)
	average_count = target_counts.mean()
	print(f'The average number of examples per label is {average_count}, threshold is {threshold_for_rare * average_count}')
	mask_rare_labels = target_counts[target_counts < threshold_for_rare * average_count]
	
	list_of_df_with_rare_labels = [Y_df]
	
	for rare_label in mask_rare_labels.index.tolist():
		df_rare_label = Y_df[Y_df[str(rare_label)] == 1]
		nbr_copies_each_rotation = int(min(max_multiplier, threshold_for_rare * average_count / len(df_rare_label.index)))
		#nbr_copies_each_flip = math.ceil(min(max_multiplier, average_count / len(df_rare_label.index)) / 3)
		
		print(f'There are {len(df_rare_label.index)} rare examples, for the label {rare_label}')
		if len(df_rare_label.index) > min_even_if_contain_0:
			df_rare_label = df_rare_label[df_rare_label['0'] == 0]
			df_rare_label = df_rare_label[df_rare_label['25'] == 0]
		print(f'There are {len(df_rare_label.index)} rare examples after removing the ones containsing 0 and 25')
		
		df_rare_label_rotation90 = df_rare_label.copy(deep = True)
		df_rare_label_rotation90.loc[:,'ReflectRotate'] = 4
		list_of_df_with_rare_labels.extend([df_rare_label_rotation90] * nbr_copies_each_rotation)
		
		df_rare_label_rotation180 = df_rare_label.copy(deep = True)
		df_rare_label_rotation180.loc[:,'ReflectRotate'] = 5
		list_of_df_with_rare_labels.extend([df_rare_label_rotation180] * nbr_copies_each_rotation)		
		
		df_rare_label_rotation270 = df_rare_label.copy(deep = True)
		df_rare_label_rotation270.loc[:,'ReflectRotate'] = 6
		list_of_df_with_rare_labels.extend([df_rare_label_rotation270] * nbr_copies_each_rotation)
		
		print(f'Label {rare_label} has only {len(df_rare_label.index)} examples, we multiply it by {3*nbr_copies_each_rotation} by incorporating: 90, 180 and 270 degrees rotations')
	
	extended_df = pd.concat(list_of_df_with_rare_labels).reset_index(drop=True)
	print(f'There are now {len(extended_df.index)} examples')
	return extended_df

def expand_data_resample(Y_df, threshold_for_rare, max_multiplier, min_even_if_contain_0):
	target_counts = Y_df.drop(["Id", "Target", "ReflectRotate"], axis=1).sum(axis=0).sort_values(ascending=False)
	average_count = target_counts.mean()
	print(f'The average number of examples per label is {average_count}, threshold is {threshold_for_rare * average_count}')
	mask_rare_labels = target_counts[target_counts < threshold_for_rare * average_count]
	
	list_of_df_with_rare_labels = [Y_df]
	
	for rare_label in mask_rare_labels.index.tolist():
		df_rare_label = Y_df[Y_df[str(rare_label)] == 1]
		nbr_copies = int(min(max_multiplier, average_count / len(df_rare_label.index)))
		
		print(f'There are {len(df_rare_label.index)} rare examples, for the label {rare_label}')
		if len(df_rare_label.index) > min_even_if_contain_0:
			df_rare_label = df_rare_label[df_rare_label['0'] == 0]
			df_rare_label = df_rare_label[df_rare_label['25'] == 0]
		print(f'There are {len(df_rare_label.index)} rare examples after removing the ones containsing 0 and 25')
		
		
		list_of_df_with_rare_labels.extend([df_rare_label] * (nbr_copies-1))
		print(f'Label {rare_label} has only {len(df_rare_label.index)} examples, we copy all of them {nbr_copies} times')
		
	extended_df = pd.concat(list_of_df_with_rare_labels).reset_index(drop=True)
	print(f'There are now {len(extended_df.index)} examples')
	return extended_df


####################################################
# Visualization of the Data
####################################################
#y_df = pd.read_csv(os.path.join(path_data_FOLDER, labelname))
y_df_only_number = pd.read_csv(os.path.join(path_data_FOLDER, labelname))
# change in one-hot encoding
for i in range(28):
    #y_df[f'{organelle_dict[i]}'] = y_df['Target'].map(lambda x: 1 if str(i) in x.strip().split() else 0)
    y_df_only_number[f'{i}'] = y_df_only_number['Target'].map(lambda x: 1 if str(i) in x.strip().split() else 0)
target_counts = y_df_only_number.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
nbr_classes_per_image = y_df_only_number.drop(["Id", "Target"], axis=1).sum(axis=1)


# =============================================================================
# with zf.ZipFile(os.path.join(path_data_FOLDER, zipname), 'r') as zip_train:
#     img_names = zip_train.namelist()
# 	
# 	# Some preliminary info on labels
#     print('In average, there are {} labels per image'.format(target_counts.sum() / y_df_only_number['Id'].size))
#     print('There are {} images without any label'.format(y_df_only_number['Target'].eq('').sum()))
#     
#     # Nbr of labels per image
#     plt.figure(figsize=(10,8))
#     plt.title('Distribution of the number of labels per image:')
#     sns.barplot(x=nbr_classes_per_image.value_counts().index, y=nbr_classes_per_image.value_counts()) #, order=target_counts.index)
#     plt.show()
# 	
# 	# Plotting a small sample
#     nbr_pics_to_plot = 2
#     plot_from_zip(zip_train, y_df_only_number, y_df_only_number['Id'].sample(n=nbr_pics_to_plot).values)
#     #visualize_part(y_df_only_number)
# =============================================================================
	
# Distribution of labels
plt.figure(figsize=(10,8))
plt.title('Graphical distribution of the labels:')
sns.barplot(x=target_counts.index.values, y=target_counts.values) #, order=target_counts.index)
plt.show()
print(f'Here is the distribution of labels: \n {target_counts}')
    
# Expand the data set parameters 
max_multiplier = 2 # do not multiply the data more times than this factor =[10, 15, ]
min_even_if_contain_0_label_1 = 25 # expand the data even if it contains the label 0 (there are SO many, we want to avoid duplicating the ones containing 0)
min_even_if_contain_0_label_2 = 100 # expand the data even if it contains the label 0 (there are SO many, we want to avoid duplicating the ones containing 0)
min_even_if_contain_0_label_resample = 750 # expand the data even if it contains the label 0 (there are SO many, we want to avoid duplicating the ones containing 0)
threshold_for_expanding_label_1 = 0.25 #if some label is represented less than this percentage, expand it
threshold_for_expanding_label_2 = 0.5 #if some label is represented less than this percentage, expand it
threshold_for_expanding_label_resample = 0.45 #if some label is represented less than this percentage, expand it
	
y_df_only_number['ReflectRotate'] = 0
	
# Expand by adding horizontal, vertical, horizontal+vertical flips
y_df_only_number = expand_data_reflections(y_df_only_number, threshold_for_expanding_label_1, max_multiplier, min_even_if_contain_0_label_1)
target_counts = y_df_only_number.drop(["Id", "Target", "ReflectRotate"], axis=1).sum(axis=0).sort_values(ascending=False)

plt.figure(figsize=(10,8))
plt.title('Graphical distribution of the labels after adding reflections:')
sns.barplot(x=target_counts.index.values, y=target_counts.values) #, order=target_counts.index)
plt.show()
	
	
	
	
	
# Expand by adding 90, 180, 270 rotations
y_df_only_number = expand_data_rotations(y_df_only_number, threshold_for_expanding_label_2, max_multiplier, min_even_if_contain_0_label_2)
target_counts = y_df_only_number.drop(["Id", "Target", "ReflectRotate"], axis=1).sum(axis=0).sort_values(ascending=False)

plt.figure(figsize=(10,8))
plt.title('Graphical distribution of the labels after adding reflections + rotations:')
sns.barplot(x=target_counts.index.values, y=target_counts.values) #, order=target_counts.index)
plt.show()
	
# Expand by resampling (just multiplying data by repetition)
y_df_only_number = expand_data_resample(y_df_only_number, threshold_for_expanding_label_resample, max_multiplier, min_even_if_contain_0_label_resample)
target_counts = y_df_only_number.drop(["Id", "Target", "ReflectRotate"], axis=1).sum(axis=0).sort_values(ascending=False)

plt.figure(figsize=(10,8))
plt.title('Graphical distribution of the labels after adding reflections + rotations + resampling:')
sns.barplot(x=target_counts.index.values, y=target_counts.values) #, order=target_counts.index)
plt.show()
	
datetime_now = str(datetime.now()).replace('-','').replace(':','').replace(' ','')
datetime_now = datetime_now[:datetime_now.find('.')]
	
new_filename = 'train_labels_extended' + datetime_now + '.csv'
y_df_only_number[['Id', 'Target', 'ReflectRotate']].to_csv(os.path.join(path_data_FOLDER, new_filename))
print(f'Saved in {os.path.join(path_data_FOLDER, new_filename)}')











