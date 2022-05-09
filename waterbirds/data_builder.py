# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to create the waterbirds datasets.

Code based on https://github.com/kohpangwei/group_DRO/blob/master/
	dataset_scripts/generate_waterbirds.py
"""
import os, shutil
import functools
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from shared import weighting as wt

DATA_DIR = '/data/ddmg/slabs/waterbirds'
IMAGE_DIR = '/data/ddmg/slabs/CUB_200_2011'
SEGMENTATION_DIR = '/data/ddmg/slabs/segmentations/'

EASY_DATA = True

NUM_PLACE_IMAGES_CLEAN = 8000
WATER_IMG_DIR_CLEAN = 'water_easy'
LAND_IMG_DIR_CLEAN = 'land_easy'

NUM_PLACE_IMAGES = 10000
WATER_IMG_DIR = 'water'
LAND_IMG_DIR = 'land'

WATERBIRD_LIST = [
	'Albatross', 'Auklet', 'Cormorant', 'Frigatebird', 'Fulmar', 'Gull', 'Jaeger',
	'Kittiwake', 'Pelican', 'Puffin', 'Tern', 'Gadwall', 'Grebe', 'Mallard',
	'Merganser', 'Guillemot', 'Pacific_Loon'
]

# water
# 'Albatross', 'Auklet',
# Fulmar Kittiwake

# land
# yellow_throated_vireo
# goldfinch



def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
	return img


def read_decode_png(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_png(img, channels=1)
	return img


def decode_number(label):
	label = tf.expand_dims(label, 0)
	label = tf.strings.to_number(label)
	return label

def map_to_image_label(x, pixel, weighted):

	bird_image = x[0]
	bird_segmentation = x[1]
	background_image = x[2]
	noise_image = x[3]

	y_list = []
	for i in range(4, x.shape[0]):
		y_list.append(x[i])

	if ((weighted == 'True') or (weighted == 'True_bal')):
		y_list.pop()
		sample_weights = x[-1]

	# decode images
	bird_image = read_decode_jpg(bird_image)
	bird_segmentation = read_decode_png(bird_segmentation)
	background_image = read_decode_jpg(background_image)
	noise_image = read_decode_png(noise_image)

	# get binary segmentation
	bird_segmentation = tf.math.round(bird_segmentation / 255)
	bird_segmentation = tf.cast(bird_segmentation, tf.uint8)

	# get binary noise
	noise_image = tf.image.resize(noise_image, (pixel, pixel))
	noise_image = tf.squeeze(noise_image)
	noise_image = tf.stack([noise_image] * 3, axis = 2)

	# resize the background image
	bkgrd_resized = tf.image.resize(background_image,
		(tf.shape(bird_image)[0], tf.shape(bird_image)[1]))
	bkgrd_resized = tf.cast(bkgrd_resized, tf.uint8)

	# get the masked image
	img = bird_image * bird_segmentation + bkgrd_resized * (1 - bird_segmentation)

	# resize
	img = tf.image.resize(img, (pixel, pixel))


	# add noise image
	img = (1 - noise_image) * img + noise_image * 3

	# rescale
	img = img / 255

	# get the label vector
	y_list_dec = [decode_number(yi) for yi in y_list]
	labels = tf.concat(y_list_dec, axis=0)

	if ((weighted == 'True') or (weighted == 'True_bal')):
		sample_weights = decode_number(sample_weights)
	else:
		sample_weights = None

	labels_and_weights = {'labels': labels, 'sample_weights': sample_weights}
	return img, labels_and_weights



def map_to_image_label_test(x, pixel, weighted):
	""" same as normal function by makes sure to not use
	weighting. """

	bird_image = x[0]
	bird_segmentation = x[1]
	background_image = x[2]
	noise_image = x[3]

	y_list = []
	for i in range(4, x.shape[0]):
		y_list.append(x[i])

	# decode images
	bird_image = read_decode_jpg(bird_image)
	bird_segmentation = read_decode_png(bird_segmentation)
	background_image = read_decode_jpg(background_image)
	noise_image = read_decode_png(noise_image)

	# get binary segmentation
	bird_segmentation = tf.math.round(bird_segmentation / 255)
	bird_segmentation = tf.cast(bird_segmentation, tf.uint8)

	# get binary noise
	noise_image = tf.image.resize(noise_image, (pixel, pixel))
	noise_image = tf.squeeze(noise_image)
	noise_image = tf.stack([noise_image] * 3, axis = 2)

	# resize the background image
	bkgrd_resized = tf.image.resize(background_image,
		(tf.shape(bird_image)[0], tf.shape(bird_image)[1]))
	bkgrd_resized = tf.cast(bkgrd_resized, tf.uint8)

	# get the masked image
	img = bird_image * bird_segmentation + bkgrd_resized * (1 - bird_segmentation)

	# resize
	img = tf.image.resize(img, (pixel, pixel))


	# add noise image
	img = (1 - noise_image) * img + noise_image * 3

	# rescale
	img = img / 255

	# get the label vector
	y_list_dec = [decode_number(yi) for yi in y_list]
	labels = tf.concat(y_list_dec, axis=0)

	if ((weighted == 'True') or (weighted == 'True_bal')):
		sample_weights = tf.ones_like(y_list_dec[0])
	else:
		sample_weights = None

	labels_and_weights = {'labels': labels, 'sample_weights': sample_weights}
	return img, labels_and_weights

def get_bird_type(x):
	bird_type = [
		water_bird_name in x['img_filename'] for water_bird_name in WATERBIRD_LIST
	]
	bird_type = max(bird_type) * 1
	return bird_type


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def create_images_labels(bird_data_frame, water_images, land_images,
	clean_back='False', rng=None):

	if clean_back == 'True':
		water_img_dir = WATER_IMG_DIR_CLEAN
		land_img_dir = LAND_IMG_DIR_CLEAN
		num_place_images = NUM_PLACE_IMAGES_CLEAN
	else:
		water_img_dir = WATER_IMG_DIR
		land_img_dir = LAND_IMG_DIR
		num_place_images = NUM_PLACE_IMAGES

	# -- randomly pick land and water images
	water_image_ids = rng.choice(water_images,
		size=int(bird_data_frame.y1.sum()), replace=False)
	water_backgrounds = [
		f'{water_img_dir}/image_{img_id}.jpg' for img_id in water_image_ids
	]

	land_image_ids = rng.choice(land_images,
		size=int((1 - bird_data_frame.y1).sum()), replace=False)
	land_backgrounds = [
		f'{land_img_dir}/image_{img_id}.jpg' for img_id in land_image_ids
	]

	bird_data_frame['background_filename'] = ''
	bird_data_frame.background_filename[(
		bird_data_frame.y1 == 1)] = water_backgrounds
	bird_data_frame.background_filename[(
		bird_data_frame.y1 == 0)] = land_backgrounds

	return bird_data_frame, water_image_ids, land_image_ids


def save_created_data(data_frame, experiment_directory, filename):
	data_frame['img_filename'] = data_frame['img_filename'].str[:-3]
	txt_df = f'{IMAGE_DIR}/images/' + data_frame.img_filename + 'jpg' + \
		',' + SEGMENTATION_DIR + data_frame.img_filename + 'png' + \
		',' + f'{DATA_DIR}/places_data/' + data_frame.background_filename + \
		',' + data_frame.noise_img

	label_cols = [
		col for col in data_frame.columns if col.startswith('y')
	]

	for colid, col in enumerate(label_cols):
		assert f'y{colid}' == col
		txt_df = txt_df + ',' + data_frame[col].astype(str)

	txt_df.to_csv(f'{experiment_directory}/{filename}.txt',
		index=False)


def extract_dim(data, v_dim, alg_step):
	data = data['0'].str.split(",", expand=True)
	data.columns = ['bird_img', 'bird_seg', 'back_img', 'noise_img'] + \
		[f'y{i}' for i in range(data.shape[1]-4)]

	v_to_drop = [f'y{i}' for i in range(1 + v_dim, data.shape[1]-4)]
	if alg_step != "None": 
		v_to_drop.append('y0')

	if len(v_to_drop) > 0:
		data.drop(v_to_drop, axis =1, inplace=True)

	txt_df = data.bird_img + \
		',' + data.bird_seg + \
		',' + data.back_img + \
		',' + data.noise_img

	label_cols = [
		col for col in data.columns if col.startswith('y')
	]

	for colid, col in enumerate(label_cols):
		txt_df = txt_df + ',' + data[col].astype(str)

	txt_df = txt_df.to_frame(name='0')
	return txt_df


def load_created_data(experiment_directory, weighted, v_dim, alg_step):
	if alg_step == 'None':
		return load_created_data_full(experiment_directory, weighted, v_dim, alg_step)
	else:
		return load_created_data_subsets(experiment_directory, weighted, v_dim, alg_step)


def load_created_data_subsets(experiment_directory, weighted, v_dim, alg_step):

	train_data_full = pd.read_csv(
		f'{experiment_directory}/train.txt')

	train_data_full = extract_dim(train_data_full, v_dim, alg_step)
	if weighted == 'True':
		train_data_full = wt.get_permutation_weights(train_data_full,
			'waterbirds', 'tr_consistent')
	elif weighted == 'True_bal':
		train_data_full = wt.get_permutation_weights(train_data_full,
			'waterbirds', 'bal')

	idx_dict = pickle.load(
		open(f'{experiment_directory}/first_step.pkl',
			'rb'))

	train_data = train_data_full.iloc[idx_dict['train_idx']]
	validation_data = train_data_full.iloc[idx_dict['valid_idx']]
	test_data = train_data_full.iloc[idx_dict['test_idx']]

	train_data = train_data.values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]

	validation_data = validation_data.values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	test_data = test_data.values.tolist()
	test_data = [
		tuple(test_data[i][0].split(',')) for i in range(len(test_data))
	]


	return train_data, validation_data, test_data




def load_created_data_full(experiment_directory, weighted, v_dim, alg_step):

	train_data = pd.read_csv(
		f'{experiment_directory}/train.txt')

	train_data = extract_dim(train_data, v_dim, alg_step)

	if weighted == 'True':
		train_data = wt.get_permutation_weights(train_data,
			'waterbirds', 'tr_consistent')
	elif weighted == 'True_bal':
		train_data = wt.get_permutation_weights(train_data,
			'waterbirds', 'bal')

	train_data = train_data.values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]

	validation_data = pd.read_csv(
		f'{experiment_directory}/valid.txt')

	validation_data = extract_dim(validation_data, v_dim, alg_step)

	if weighted == 'True':
		validation_data = wt.get_permutation_weights(validation_data,
			'waterbirds', 'tr_consistent')
	elif weighted == 'True_bal':
		validation_data = wt.get_permutation_weights(validation_data,
			'waterbirds', 'tr_consistent')

	validation_data = validation_data.values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	test_data_dict = {}
	for dist in [0.1, 0.5, 0.9]:
		test_data = pd.read_csv(
			f'{experiment_directory}/test_{dist}.txt'
		)
		test_data = extract_dim(test_data, v_dim, alg_step)
		test_data = test_data.values.tolist()

		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		test_data_dict[f'{dist}'] = test_data

	return train_data, validation_data, test_data_dict


def create_noise_patches(experiment_directory, df, group, rng):
	if not os.path.exists(f'{experiment_directory}/noise_imgs'):
		os.mkdir(f'{experiment_directory}/noise_imgs')

	if not os.path.exists(f'{experiment_directory}/noise_imgs/{group}'):
		os.mkdir(f'{experiment_directory}/noise_imgs/{group}')

		# this is a blanck (no noise) image. Need it for code consistency
		no_noise_img = f'{experiment_directory}/noise_imgs/{group}/no_noise.png'
		tf.keras.preprocessing.image.save_img(no_noise_img,
					tf.zeros(shape=[128, 128, 1]), scale=False
			)
	df['noise_img'] = f'{experiment_directory}/noise_imgs/{group}/no_noise.png'

	# y2_noise = df.y2.copy()
	# y2_noise_flip_idx = 	rng.choice(
	# 	range(N), size=(int(0.01 * N)), replace =False).tolist()
	# y2_noise[y2_noise_flip_idx] = 1.0 - y2_noise[y2_noise_flip_idx]
	# df['y2_noise'] =
	for i in range(df.shape[0]):
		if df.y2[i] == 1:
			# create random pixel "flips"
			noise = tf.constant(rng.binomial(n=1, p=0.0005,size=(128, 128)))
			noise = tf.reshape(noise, [128, 128,1])
			noise = tf.cast(noise, dtype=tf.float32)
			# convolution to make patches
			kernel = tf.ones(shape=[10, 10, 1, 1])
			kernel = tf.cast(kernel, dtype=tf.float32)

			noise_reshaped = tf.reshape(
					noise, [1] + tf.shape(noise).numpy().tolist())

			noise_conv = tf.nn.conv2d(
					tf.cast(noise_reshaped, dtype=tf.float32),
					kernel, [1, 1, 1, 1], padding='SAME'
			)

			noise_conv = tf.squeeze(noise_conv, axis=0)

			# just need 1/0 so threshold
			noise_conv = noise_conv >= 1.0
			noise_conv = tf.cast(noise_conv, dtype=tf.float32)

			# save
			bird_name = df.img_filename.iloc[i].split('/')[-1][:-4]
			bird_noise_img = f'{experiment_directory}/noise_imgs/{group}/{bird_name}.png'
			tf.keras.preprocessing.image.save_img(bird_noise_img,
					tf.reshape(noise_conv, [128, 128, 1]), scale=False
			)
			df.noise_img.iloc[i] = bird_noise_img
	return df


def get_simulated_labels(df, py0, ideal, reverse, rng):

	N = df.shape[0]
	y1 = rng.binomial(1, 0.5, (N, 1))
	y2 = y1.copy()
	y2_flip_idx = rng.choice(
		range(N), size=(int(0.3 * N)), replace=False).tolist()
	y2[y2_flip_idx] = 1.0 - y2[y2_flip_idx]

	# --- create redundant aux labels
	D2 = 10
	y_other = rng.binomial(1, 0.01, size=(N, D2))

	# --- create final label
	coef_uy0 = np.array([[0.84], [0.4]])
	y0_bias = -0.84
	if ideal:
		coef_uy0 = np.zeros_like(coef_uy0)
		y0_bias = -0.15
	if reverse:
		coef_uy0 = np.array([[-0.84], [-0.4]])
		y0_bias = 0.45

	coef_uyother = rng.normal(0, 1, (y_other.shape[1], 1))

	y0 = y0_bias + np.dot(np.hstack([y1, y2]), coef_uy0) + np.dot(y_other,
			coef_uyother)
	y0 = y0 + rng.normal(0, 0.5, (N, 1))
	y0 = (sigmoid(y0) > 0.5) * 1.0
	# --- merge the created label data with the bird data
	label_df = pd.DataFrame(y_other)
	label_df.columns = [f'y{i}' for i in range(3, D2 + 3)]
	label_df['y0'] = y0
	label_df['y1'] = y1
	label_df['y2'] = y2

	# print(label_df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
	# 	["mean", "count"]).reset_index())
	label_df['idx'] = label_df.groupby(['y0']).cumcount()
	df['idx'] = df.groupby(['y0']).cumcount()
	df = df.merge(label_df, on=['idx', 'y0'])
	df.drop('idx', axis=1, inplace=True)
	df.reset_index(inplace=True, drop=True)

	# print("===before fixing marginal====")
	# print(df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
	# 	["mean", "count"]).reset_index())

	# print(df.y0.value_counts(normalize=True))

	if ideal | reverse:
		df = fix_marginal(df, y0_probability=py0, rng=rng)

	# print("=====after fixing marginal======")
	print(df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
		["mean", "count"]).reset_index())

	# print(df[[f'y{i}' for i in range(13)]].groupby(
	# [f'y{i}' for i in range(1, 13)]).agg(
	#     ["mean", "count"]).reset_index())

	# print(df.y0.value_counts(normalize=True))

	df = df[['img_filename'] + [f'y{i}' for i in range(df.shape[1] - 1)]]
	df.reset_index(inplace=True, drop=True)
	return df


def create_splits(experiment_directory, random_seed): 
	rng = np.random.RandomState(random_seed + 1234)
	train_data = pd.read_csv(
		f'{experiment_directory}/train.txt')

	# --- split into training validation and estimation/testing
	train_valid_idx = rng.choice(train_data.shape[0], 
		size = int(0.5*train_data.shape[0]), 
		replace = False).tolist()

	test_idx = list(
		set(range(train_data.shape[0])) - set(train_valid_idx))

	# --- split into first step train and test 
	train_idx = rng.choice(train_valid_idx, 
		size = int(0.7 * len(train_valid_idx)))

	valid_idx = list(
		set(range(len(train_valid_idx))) - set(train_idx))

	idx_dict = {
		'train_idx': train_idx, 
		'valid_idx': valid_idx, 
		'test_idx': test_idx
	}


	# train_data = train_data['0'].str.split(",", expand=True)
	# train_data.columns = ['bird_img', 'bird_seg', 'back_img', 'noise_img'] + \
	# 	[f'y{i}' for i in range(train_data.shape[1]-4)]

	# train_data.drop(['bird_img', 'bird_seg', 'back_img', 'noise_img'], 
	# 	axis=1, inplace=True)
	# train_data = train_data.astype(float)
	# step_one_train_data = train_data.iloc[train_idx]
	# step_one_valid_data = train_data.iloc[valid_idx]
	# step_two_train_data = train_data.iloc[test_idx]


	# print("====step 1 train ========")
	# print(step_one_train_data.mean(axis=0))
	# print("=====step 1 valid =========")
	# print(step_one_valid_data.mean(axis=0))
	# print("====== step 2 ======")
	# print(step_two_train_data.mean(axis=0))

	pickle.dump(idx_dict, open(
		f'{experiment_directory}/first_step.pkl', 'wb'))




def create_save_waterbird_lists(experiment_directory, v_dim,
	p_tr=0.7, p_val = 0.25,
	clean_back='False', random_seed=None):

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	if clean_back == 'True':
		num_place_images = NUM_PLACE_IMAGES_CLEAN
	else:
		num_place_images = NUM_PLACE_IMAGES

	# --- read in all bird image filenames
	df = pd.read_csv(f'{IMAGE_DIR}/images.txt', sep=" ", header=None,
		names=['img_id', 'img_filename'], index_col='img_id')
	df = df.sample(frac=1, random_state=random_seed)
	df.reset_index(inplace=True, drop=True)

	if EASY_DATA:
		# df = df[((
		# 	df.img_filename.str.contains('Gull')) | (
		# 	df.img_filename.str.contains('Warbler')
		# ))]
		df['bird_name'] = df.img_filename.str.split("/", expand=True)[0].str.split(".", expand=True)[1].str.lower()
		df = df[(
			(df.bird_name.str.contains('gull')) |
			(df.bird_name.str.contains('warbler')) |
			(df.bird_name.str.contains('vireo')) |
			(df.bird_name.str.contains('common_yellowthroat')) |
			(df.bird_name.str.contains('american_redstart')) |
			(df.bird_name.str.contains('ovenbird')) |
			(df.bird_name.str.contains('yellow_breasted_chat')) |
			(df.bird_name.str.contains('northern_waterthrush')) |
			(df.bird_name.str.contains('louisiana_waterthrush')) |
			(df.bird_name.str.contains('albatross')) |
			(df.bird_name.str.contains('auklet'))
			)]

		df.drop('bird_name', axis=1, inplace=True)
		df.reset_index(inplace=True, drop=True)

	# -- get bird type
	df['y0'] = df.apply(get_bird_type, axis=1)

	# -- split into train and test
	train_val_ids = rng.choice(df.shape[0],
		size=int(p_tr * df.shape[0]), replace=False).tolist()
	df['train_valid_ids'] = 0
	df.train_valid_ids.loc[train_val_ids] = 1

	# ------------------------------------------- #
	# --- get the train and validation data ----- #
	# ------------------------------------------- #
	train_valid_df = df[(df.train_valid_ids == 1)].reset_index(drop=True)
	train_valid_df.drop(['train_valid_ids'], axis=1, inplace=True)
	train_valid_df = get_simulated_labels(
		df=train_valid_df,
		py0=None,
		ideal=False,
		reverse=False,
		rng=rng)

	# ---- generate noise images
	train_valid_df = create_noise_patches(experiment_directory, train_valid_df,
	'train_valid', rng)

	train_valid_df, used_water_img_ids, used_land_img_ids = create_images_labels(
		train_valid_df, num_place_images, num_place_images,
		clean_back=clean_back, rng=rng)

	train_ids = rng.choice(train_valid_df.shape[0],
		size=int((1.0 - p_val) * train_valid_df.shape[0]), replace=False).tolist()
	train_valid_df['train'] = 0
	train_valid_df.train.loc[train_ids] = 1


	# --- get the marginal distribution
	fixed_marginal = 1.0 - train_valid_df.y0.mean()
	# --- save training data
	train_df = train_valid_df[(train_valid_df.train == 1)].reset_index(drop=True)
	# print(train_df[['y0', 'y1', 'y2']].groupby(['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(train_df.shape[0]), size = int(0.05 * train_df.shape[0]), replace=False).tolist()
	train_df['y0'].iloc[flip_label] = 1.0 - train_df['y0'].iloc[flip_label]


	save_created_data(train_df, experiment_directory=experiment_directory,
		filename='train')

	# --- save validation data
	valid_df = train_valid_df[(train_valid_df.train == 0)].reset_index(drop=True)
	# print(valid_df[['y0', 'y1', 'y2']].groupby(['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(valid_df.shape[0]), size = int(0.05 * valid_df.shape[0]), replace=False).tolist()
	valid_df['y0'].iloc[flip_label] = 1.0 - valid_df['y0'].iloc[flip_label]

	save_created_data(valid_df, experiment_directory=experiment_directory,
		filename='valid')

	# ------------------------------------------- #
	# -------------- get the test data ---------- #
	# ------------------------------------------- #

	test_df = df[(df.train_valid_ids == 0)].reset_index(drop=True)
	test_df.drop(['train_valid_ids'], axis=1, inplace=True)

	available_water_ids = list(set(range(num_place_images)) - set(
		used_water_img_ids))
	available_land_ids = list(set(range(num_place_images)) - set(
		used_land_img_ids))

	# in dist
	curr_test_df = test_df.copy()
	print(curr_test_df.head())

	curr_test_df = get_simulated_labels(
		df=curr_test_df,
		py0=None,
		ideal=False,
		reverse=False,
		rng=rng)

	curr_test_df = create_noise_patches(experiment_directory, curr_test_df,
		'test_0.9', rng)
	curr_test_df, _, _ = create_images_labels(
		curr_test_df, available_water_ids, available_land_ids,
		clean_back=clean_back, rng=rng)
	# print(curr_test_df[['y0', 'y1', 'y2']].groupby(
	# 	['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y0'].iloc[flip_label] = 1.0 - curr_test_df['y0'].iloc[flip_label]

	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename='test_0.9')

	# ood1
	curr_test_df = test_df.copy()
	curr_test_df = get_simulated_labels(
		df=curr_test_df,
		py0=fixed_marginal,
		ideal=True,
		reverse=False,
		rng=rng)
	curr_test_df = create_noise_patches(experiment_directory, curr_test_df,
		'test_0.5', rng)
	curr_test_df, _, _ = create_images_labels(
		curr_test_df, available_water_ids, available_land_ids,
		clean_back=clean_back, rng=rng)
	# print(curr_test_df[['y0', 'y1', 'y2']].groupby(
	# 	['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y0'].iloc[flip_label] = 1.0 - curr_test_df['y0'].iloc[flip_label]

	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename='test_0.5')

	# ood2
	curr_test_df = test_df.copy()
	curr_test_df = get_simulated_labels(
		df=curr_test_df,
		py0=fixed_marginal,
		ideal=False,
		reverse=True,
		rng=rng)
	curr_test_df = create_noise_patches(experiment_directory, curr_test_df,
		'test_0.1', rng)
	curr_test_df, _, _ = create_images_labels(
		curr_test_df, available_water_ids, available_land_ids,
		clean_back=clean_back, rng=rng)
	# print(curr_test_df[['y0', 'y1', 'y2']].groupby(
	# 	['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y0'].iloc[flip_label] = 1.0 - curr_test_df['y0'].iloc[flip_label]

	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename='test_0.1')


def fix_marginal(df, y0_probability, rng):
	y0_group = df.index[(df.y0 == 0)]
	y1_group = df.index[(df.y0 == 1)]

	y1_probability = 1 - y0_probability
	if len(y0_group) < (y0_probability / y1_probability) * len(y1_group):
		y0_ids = deepcopy(y0_group).tolist()
		y1_ids = rng.choice(
			y1_group, size=int((y1_probability / y0_probability) * len(y0_group)),
			replace=False).tolist()
	elif len(y1_group) < (y1_probability / y0_probability) * len(y0_group):
		y1_ids = deepcopy(y1_group).tolist()
		y0_ids = rng.choice(
			y0_group, size=int((y0_probability / y1_probability) * len(y1_group)),
			replace=False
		).tolist()
	dff = df.iloc[y1_ids + y0_ids]
	dff.reset_index(inplace=True, drop=True)
	reshuffled_ids = rng.choice(dff.index,
		size=len(dff.index), replace=False).tolist()
	dff = dff.iloc[reshuffled_ids].reset_index(drop=True)
	return dff

def build_input_fns(data_dir, weighted='False', p_tr=.7, p_val = 0.25,
 v_dim=0, clean_back='False', random_seed=None, alg_step='None'):

	experiment_directory = (f'{data_dir}/experiment_data/'
		f'rs{random_seed}')

	# --- generate splits if they dont exist
	if not os.path.exists(f'{experiment_directory}/train.txt'):
		if not os.path.exists(experiment_directory):
			os.mkdir(experiment_directory)

		create_save_waterbird_lists(
			experiment_directory=experiment_directory,
			v_dim=v_dim,
			p_tr=p_tr,
			p_val=p_val,
			clean_back=clean_back,
			random_seed=random_seed)


	if not os.path.exists(f'{experiment_directory}/first_step.pkl'):
		create_splits(experiment_directory, random_seed)

	# --load splits
	train_data, valid_data, test_data_dict = load_created_data(
		experiment_directory=experiment_directory, weighted=weighted,
		v_dim=v_dim, alg_step=alg_step)

	# --this helps auto-set training steps at train time
	training_data_size = len(train_data)

	# Build an iterator over training batches.
	def train_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'], weighted=params['weighted'])
		batch_size = params['batch_size']
		num_epochs = params['num_epochs']

		dataset = tf.data.Dataset.from_tensor_slices(train_data)
		dataset = dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
		# dataset = dataset.shuffle(int(1e5), seed=random_seed).batch(batch_size).repeat(num_epochs)
		dataset = dataset.batch(batch_size).repeat(num_epochs)
		return dataset

	# Build an iterator over validation batches

	def valid_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'], weighted=params['weighted'])
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.map(map_to_image_label_given_pixel,
			num_parallel_calls=1)
		valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
		return valid_dataset

	# build an iterator over whole validation dataset
	def final_valid_input_fn(params):
		map_to_image_label_given_pixel = functools.partial(map_to_image_label,
			pixel=params['pixel'], weighted=params['weighted'])
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.map(map_to_image_label_given_pixel,
			num_parallel_calls=1)
		valid_dataset = valid_dataset.batch(int(1e5)).repeat(1)
		return valid_dataset

	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params, fixed_joint=False, aux_joint_skew=0.5):
		del fixed_joint, aux_joint_skew
		map_to_image_label_given_pixel = functools.partial(map_to_image_label_test,
			pixel=params['pixel'], weighted=params['weighted'])

		shifted_test_data = test_data_dict[f'{py}']

		def eval_input_fn():
			eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
			eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
			eval_shift_dataset = eval_shift_dataset.batch(int(1e5)).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, final_valid_input_fn, eval_input_fn_creater
