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

	if weighted == 'True':
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

	if weighted == 'True':
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

	if weighted == 'True':
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


def extract_dim(data, v_dim):
	data = data['0'].str.split(",", expand=True)
	data.columns = ['bird_img', 'bird_seg', 'back_img', 'noise_img'] + \
		[f'y{i}' for i in range(data.shape[1]-4)]

	v_to_drop = [f'y{i}' for i in range(3 + v_dim, data.shape[1]-4)]
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
		assert f'y{colid}' == col
		txt_df = txt_df + ',' + data[col].astype(str)

	txt_df = txt_df.to_frame(name='0')
	return txt_df

def load_created_data(experiment_directory, weighted, v_dim,
	alg_step):

	train_data = pd.read_csv(
		f'{experiment_directory}/train.txt')

	train_data = extract_dim(train_data, v_dim)

	if weighted == 'True':
		train_data = wt.get_binary_weights(train_data,
			'waterbirds')

	train_data = train_data.values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]
	validation_data = pd.read_csv(
		f'{experiment_directory}/valid.txt')

	validation_data = extract_dim(validation_data, v_dim)

	if weighted == 'True':
		validation_data = wt.get_binary_weights(validation_data,
			'waterbirds')

	validation_data = validation_data.values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	if alg_step == 'first':
		first_second_step_idx = pickle.load(
			open(f'{experiment_directory}/first_second_step_idx.pkl', 'rb'))
		train_idx = first_second_step_idx['first']['train_idx']
		valid_idx = first_second_step_idx['first']['valid_idx']

		validation_data = [train_data[i] for i in valid_idx]
		train_data = [train_data[i] for i in train_idx]

	elif alg_step == 'second':
		first_second_step_idx = pickle.load(
			open(f'{experiment_directory}/first_second_step_idx.pkl', 'rb'))
		train_idx = first_second_step_idx['second']['train_idx']
		train_data = [train_data[i] for i in train_idx]


	test_data_dict = {}

	test_same_data = pd.read_csv(
		f'{experiment_directory}/test_same.txt'
	)
	test_same_data = extract_dim(test_same_data, v_dim)
	test_same_data = test_same_data.values.tolist()

	test_same_data = [
		tuple(test_same_data[i][0].split(',')) for i in range(len(test_same_data))
	]
	test_data_dict['same'] = test_same_data

	test_shift_data = pd.read_csv(
		f'{experiment_directory}/test_shift.txt'
	)

	test_shift_data = extract_dim(test_shift_data, v_dim)
	test_shift_data = test_shift_data.values.tolist()

	test_shift_data = [
		tuple(test_shift_data[i][0].split(',')) for i in range(len(test_shift_data))
	]
	test_data_dict['shift'] = test_shift_data

	return train_data, validation_data, test_data_dict


def create_noise_patches(experiment_directory, df, rng):
	if not os.path.exists(f'{experiment_directory}/noise_imgs'):
		os.mkdir(f'{experiment_directory}/noise_imgs')

		# this is a blanck (no noise) image. Need it for code consistency
		no_noise_img = f'{experiment_directory}/noise_imgs/no_noise.png'
		tf.keras.preprocessing.image.save_img(no_noise_img,
			    tf.zeros(shape=[128, 128, 1]), scale=False
			)
	df['noise_img'] = f'{experiment_directory}/noise_imgs/no_noise.png'

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
			bird_noise_img = f'{experiment_directory}/noise_imgs/{bird_name}.png'
			tf.keras.preprocessing.image.save_img(bird_noise_img,
			    tf.reshape(noise_conv, [128, 128, 1]), scale=False
			)
			df.noise_img.iloc[i] = bird_noise_img
	return df



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
		df = df[((
			df.img_filename.str.contains('Gull')) | (
			df.img_filename.str.contains('Warbler')
		))]

		df.reset_index(inplace=True, drop=True)

	# -- get bird type
	df['y0'] = df.apply(get_bird_type, axis=1)


	# -- generate the relevant aux labels
	N = df.shape[0]
	g = rng.binomial(n=1, p=0.3, size=(N, 1))

	D1 = 2
	u = g * rng.normal(1, 1, size=(N,D1)) + \
    (1.0 - g) * rng.normal(-1, 1, size=(N,D1))

	coef_uy1 = rng.normal(0.5, 0.1, size=(D1, 1))
	coef_uy2 = rng.normal(0.5, 0.1, size=(D1, 1))

	y1 = (sigmoid(np.dot(u, coef_uy1)) > 0.5) * 1.0
	y1_flip_idx =rng.choice(
    range(N), size=(int(0.05 * N)), replace =False).tolist()
	y1[y1_flip_idx] = 1.0 - y1[y1_flip_idx]

	y2 = (sigmoid(np.dot(u, coef_uy2)) > 0.5) * 1.0
	y2_flip_idx =rng.choice(
    range(N), size=(int(0.05 * N)), replace =False).tolist()
	y2[y2_flip_idx] = 1.0 - y2[y2_flip_idx]

	# ---- create redundant aux labels
	if v_dim >0:
		y_other = rng.binomial(1, 1.0/v_dim, size=(N, v_dim))
	else:
		y_other = np.zeros(shape=(N, 1))

	# --- create final label
	coef_uy0 = rng.normal(1, 0.1, (u.shape[1], 1))
	coef_uyother = rng.normal(-1, 0.1, (y_other.shape[1], 1))

	y0 = np.dot(u, coef_uy0)
	if v_dim > 0:
		y0 = y0 + np.dot(y_other, coef_uyother)
	y0 = (sigmoid(y0) > 0.5)*1.0

	# --- merge the created label data with the bird data
	if v_dim > 0:
		label_df = pd.DataFrame(y_other)
		label_df.columns = [f'y{i}' for i in range(3, v_dim+3)]
		label_df['y0'] = y0
		label_df['y1'] = y1
		label_df['y2'] = y2
	else:
		label_df = pd.DataFrame({
			'y0': y0[:, 0],
			'y1': y1[:, 0],
			'y2': y2[:, 0]
			})

	print(label_df.head())
	label_df['idx'] = label_df.groupby(['y0']).cumcount()
	df['idx'] = df.groupby(['y0']).cumcount()
	df = df.merge(label_df, on = ['idx', 'y0'])

	df.drop('idx', axis=1, inplace=True)

	df = df[['img_filename'] + [f'y{i}' for i in range(df.shape[1] - 1)]]
	df.reset_index(inplace =True, drop=True)
	# ---- generate noise images
	df = create_noise_patches(experiment_directory, df, rng)

	train_val_ids = rng.choice(df.shape[0],
		size=int(p_tr * df.shape[0]), replace=False).tolist()
	df['train_valid_ids'] = 0
	df.train_valid_ids.loc[train_val_ids] = 1

	# --- get the train and validation data
	train_valid_df = df[(df.train_valid_ids == 1)].reset_index(drop=True)

	train_valid_df, used_water_img_ids, used_land_img_ids = create_images_labels(
		train_valid_df, num_place_images, num_place_images,
		clean_back=clean_back, rng=rng)

	# --- create train validation split
	train_ids = rng.choice(train_valid_df.shape[0],
		size=int((1.0 - p_val) * train_valid_df.shape[0]), replace=False).tolist()
	train_valid_df['train'] = 0
	train_valid_df.train.loc[train_ids] = 1

	# --- save training data
	train_df = train_valid_df[(train_valid_df.train == 1)].reset_index(drop=True)
	save_created_data(train_df, experiment_directory=experiment_directory,
		filename='train')

	# --- save validation data
	valid_df = train_valid_df[(train_valid_df.train == 0)].reset_index(drop=True)
	save_created_data(valid_df, experiment_directory=experiment_directory,
		filename='valid')

	# --- create + save test data
	test_df = df[(df.train_valid_ids == 0)].reset_index(drop=True)

	available_water_ids = list(set(range(num_place_images)) - set(
		used_water_img_ids))
	available_land_ids = list(set(range(num_place_images)) - set(
		used_land_img_ids))

	# in dist
	curr_test_df = test_df.copy()
	curr_test_df, _, _ = create_images_labels(
		curr_test_df, available_water_ids, available_land_ids,
		 clean_back=clean_back, rng=rng)
	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename=f'test_same')

	# ood
	curr_test_df = test_df.copy()
	py0 = np.mean(curr_test_df.y0)
	smallest_group = min(
		curr_test_df[['y1', 'y2']].groupby(['y1', 'y2']).size().reset_index()[0]
		)

	curr_test_df = curr_test_df.groupby(
		['y1', 'y2']).sample(n=smallest_group).reset_index(drop=True)
	curr_test_df = fix_marginal(curr_test_df, py0, rng)
	curr_test_df, _, _ = create_images_labels(
		curr_test_df, available_water_ids, available_land_ids,
		 clean_back=clean_back, rng=rng)
	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename=f'test_shift')




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
		f'rs{random_seed}_v_dim{v_dim}')

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
		dataset = dataset.shuffle(int(1e5)).batch(batch_size).repeat(num_epochs)
		# dataset = dataset.batch(batch_size).repeat(num_epochs)
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

	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params, fixed_joint=False, aux_joint_skew=0.5):
		del fixed_joint, aux_joint_skew
		map_to_image_label_given_pixel = functools.partial(map_to_image_label_test,
			pixel=params['pixel'], weighted=params['weighted'])

		dist = 'same' if py==0.9 else 'shift'
		shifted_test_data = test_data_dict[dist]
		batch_size = params['batch_size']

		def eval_input_fn():
			eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
			eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_given_pixel)
			eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, eval_input_fn_creater