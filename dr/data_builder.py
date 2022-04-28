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

"""Functions to create the diabetic retionpathy datasets."""
import os
import functools
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from shared import weighting as wt

IMAGES_DIR = '/data/ddmg/scate/dr'

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
	fundus_image = x[0]
	noise_image = x[1]
	y0 = x[2]
	y1 = x[3]

	if ((weighted == 'True') or (weighted == 'True_bal')):
		sample_weights = x[-1]

	# decode images
	img = read_decode_jpg(fundus_image)
	img = tf.cast(img, tf.float32)

	noise_image = read_decode_png(noise_image)

	# get binary noise
	noise_image = tf.image.resize(noise_image, (pixel, pixel))
	noise_image = tf.squeeze(noise_image)
	noise_image = tf.stack([noise_image] * 3, axis=2)

	# add noise image
	img = (1 - noise_image) * img + noise_image

	# rescale
	img = img / 255

	# get the label vector
	y0 = decode_number(y0)
	y1 = decode_number(y1)
	labels = tf.concat([y0, y1], axis=0)

	if ((weighted == 'True') or (weighted == 'True_bal')):
		sample_weights = decode_number(sample_weights)
	else:
		sample_weights = None

	labels_and_weights = {'labels': labels, 'sample_weights': sample_weights}
	return img, labels_and_weights


def map_to_image_label_test(x, pixel, weighted):
	fundus_image = x[0]
	noise_image = x[1]
	y0 = x[2]
	y1 = x[3]

	# decode images
	img = read_decode_jpg(fundus_image)
	img = tf.cast(img, tf.float32)
	noise_image = read_decode_png(noise_image)

	# get binary noise
	noise_image = tf.image.resize(noise_image, (pixel, pixel))
	noise_image = tf.squeeze(noise_image)
	noise_image = tf.stack([noise_image] * 3, axis=2)

	# add noise image
	img = (1 - noise_image) * img + noise_image

	# rescale
	img = img / 255

	# get the label vector
	y0 = decode_number(y0)
	y1 = decode_number(y1)
	labels = tf.concat([y0, y1], axis=0)

	if ((weighted == 'True') or (weighted == 'True_bal')):
		sample_weights = tf.ones_like(y0)
	else:
		sample_weights = None

	labels_and_weights = {'labels': labels, 'sample_weights': sample_weights}
	return img, labels_and_weights

def save_created_data(data_frame, experiment_directory, filename):
	txt_df = f'{IMAGES_DIR}/images_processed/' + \
		data_frame.img_filename + '.jpg' + \
		',' + data_frame.noise_img

	label_cols = [
		col for col in data_frame.columns if col.startswith('y')
	]

	for colid, col in enumerate(label_cols):
		assert f'y{colid}' == col
		txt_df = txt_df + ',' + data_frame[col].astype(str)

	txt_df.to_csv(f'{experiment_directory}/{filename}.txt',
		index=False)


def load_created_data(experiment_directory, weighted,
	alg_step):

	train_data = pd.read_csv(
		f'{experiment_directory}/train.txt')

	if weighted == 'True':
		train_data = wt.get_permutation_weights(train_data,
			'dr', 'tr_consistent')
	elif weighted == 'True_bal':
		raise NotImplementedError("not yet")

	train_data = train_data.values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]
	validation_data = pd.read_csv(
		f'{experiment_directory}/valid.txt')

	if weighted == 'True':
		validation_data = wt.get_permutation_weights(validation_data,
			'dr', 'tr_consistent')
	elif weighted == 'True_bal':
		raise NotImplementedError("not yet")

	validation_data = validation_data.values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	if alg_step == 'first':
		raise NotImplementedError("not yet")
	elif alg_step == 'second':
		raise NotImplementedError("not yet")

	test_data_dict = {}

	for py1y0_s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
		test_data = pd.read_csv(
			f'{experiment_directory}/test_{py1y0_s}.txt'
		)

		test_data = test_data.values.tolist()

		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		test_data_dict[f'{py1y0_s}'] = test_data

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

	for i in range(df.shape[0]):
		if df.y1[i] == 1:
			# create random pixel "flips"
			noise = tf.constant(rng.binomial(n=1, p=0.0005,
				size=(128, 128)))
			noise = tf.reshape(noise, [128, 128, 1])
			noise = tf.cast(noise, dtype=tf.float32)
			# convolution to make patches
			kernel = tf.ones(shape=[7, 7, 1, 1])
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
			patient_eye_id = df.img_filename.iloc[i]
			bird_noise_img = f'{experiment_directory}/noise_imgs/{group}/{patient_eye_id}.png'
			tf.keras.preprocessing.image.save_img(bird_noise_img,
				tf.reshape(noise_conv, [128, 128, 1]), scale=False
			)
			df.noise_img.iloc[i] = bird_noise_img
	return df


def get_simulated_labels(df, py1y0, rng):
	df['y1'] = 0.0
	all_indices = []
	for y0_val in df.y0.unique():
		temp_df = df[(df.y0 == y0_val)]
		if y0_val == 0:
			indices = rng.choice(temp_df.index.tolist(),
				size=int(py1y0 * temp_df.shape[0]), replace=False).tolist()
		else:
			indices = rng.choice(temp_df.index.tolist(),
				size=int((1.0 - py1y0) * temp_df.shape[0]),
				replace=False).tolist()

		all_indices = all_indices + indices

	df.loc[all_indices, 'y1'] = 1.
	df.reset_index(inplace=True, drop=True)
	return df


def create_save_dr_lists(experiment_directory, py1y0,
	p_tr=0.7, p_val=0.25, random_seed=None):

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	# --- read in all image filenames and labels
	df = pd.read_csv(f'{IMAGES_DIR}/trainLabels.csv')
	df.rename(columns={'image': 'img_filename',
		'level':'y0'}, inplace=True)

	# --- remove all the exmples that I was not able to preprocess
	processed_images = pd.DataFrame([
		f[:-4] for f in os.listdir(f'{IMAGES_DIR}/images_processed')
	])
	processed_images.columns = ['img_filename']
	df = df.merge(processed_images, on=['img_filename'])

	df['y0'] = df.y0.astype(float)
	df = df.sample(frac=1, random_state=random_seed)
	df.reset_index(inplace=True, drop=True)

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
	train_valid_df = get_simulated_labels(train_valid_df,
		py1y0=py1y0, rng=rng)

	# ---- generate noise images
	train_valid_df = create_noise_patches(experiment_directory, train_valid_df,
	'train_valid', rng)

	train_ids = rng.choice(train_valid_df.shape[0],
		size=int((1.0 - p_val) * train_valid_df.shape[0]), replace=False).tolist()
	train_valid_df['train'] = 0
	train_valid_df.train.loc[train_ids] = 1

	# --- save training data
	train_df = train_valid_df[(train_valid_df.train == 1)].reset_index(drop=True)
	flip_label = rng.choice(range(train_df.shape[0]), 
		size = int(0.05 * train_df.shape[0]), replace=False).tolist()
	train_df['y0'].iloc[flip_label] = rng.choice(range(5), 
		size=int(0.05 * train_df.shape[0]), replace=True)
	print(train_df[['y0', 'y1']].groupby(['y0', 'y1']).size().reset_index())
	print("==== marginal in train")
	print(train_df.value_counts(normalize=True))
	save_created_data(train_df, experiment_directory=experiment_directory,
		filename='train')

	# --- save validation data
	valid_df = train_valid_df[(train_valid_df.train == 0)].reset_index(drop=True)
	
	flip_label = rng.choice(range(valid_df.shape[0]), 
		size = int(0.05 * valid_df.shape[0]), replace=False).tolist()
	valid_df['y0'].iloc[flip_label] = rng.choice(range(5), 
		size=int(0.05 * valid_df.shape[0]), replace=True)
	print(valid_df[['y0', 'y1']].groupby(['y0', 'y1']).size().reset_index())
	print("==== marginal in valid")
	print(valid_df.y0.value_counts(normalize=True))
	save_created_data(valid_df, experiment_directory=experiment_directory,
		filename='valid')

	# ------------------------------------------- #
	# -------------- get the test data ---------- #
	# ------------------------------------------- #

	test_df = df[(df.train_valid_ids == 0)].reset_index(drop=True)
	test_df.drop(['train_valid_ids'], axis=1, inplace=True)

	for py1y0_s in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
		# in dist
		curr_test_df = test_df.copy()
		curr_test_df = get_simulated_labels(curr_test_df, py1y0=py1y0_s,
			rng=rng)
		curr_test_df = create_noise_patches(experiment_directory, curr_test_df,
			f'test_{py1y0_s}', rng)
		
		
		flip_label = rng.choice(range(curr_test_df.shape[0]), 
			size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
		curr_test_df['y0'].iloc[flip_label] = rng.choice(range(5), 
			size=int(0.05 * curr_test_df.shape[0]), replace=True)

		print(curr_test_df[['y0', 'y1']].groupby(['y0', 'y1']).size().reset_index())
		print(f"==== marginal in test {py1y0_s}")
		print(curr_test_df.y0.value_counts(normalize=True))
		save_created_data(curr_test_df, experiment_directory=experiment_directory,
			filename=f'test_{py1y0_s}')


def build_input_fns(data_dir, weighted='False', py1y0=0.9, p_tr=.7, p_val=0.25,
	random_seed=None, alg_step='None'):

	experiment_directory = (f'{data_dir}/experiment_data/'
		f'rs{random_seed}')

	# --- generate splits if they dont exist
	if not os.path.exists(f'{experiment_directory}/train.txt'):
		if not os.path.exists(experiment_directory):
			os.mkdir(experiment_directory)

		create_save_dr_lists(
			experiment_directory=experiment_directory,
			py1y0=py1y0,
			p_tr=p_tr,
			p_val=p_val,
			random_seed=random_seed)

	# --load splits
	train_data, valid_data, test_data_dict = load_created_data(
		experiment_directory=experiment_directory, weighted=weighted,
		alg_step=alg_step)

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
		valid_dataset = valid_dataset.batch(params['batch_size']).repeat(1)
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
			eval_shift_dataset = eval_shift_dataset.batch(params['batch_size']).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return training_data_size, train_input_fn, valid_input_fn, final_valid_input_fn, eval_input_fn_creater
