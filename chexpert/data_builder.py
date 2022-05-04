"""Functions to create the chexpert datasets."""
import os, shutil
import functools
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from shared import weighting as wt

SOTO_MAIN_DIR = '/nfs/turbo/coe-soto'
RBG_MAIN_DIR = '/nfs/turbo/coe-rbg'
MIT_MAIN_DIR = '/data/ddmg/slabs'

if os.path.isdir(SOTO_MAIN_DIR):
	MAIN_DIR = SOTO_MAIN_DIR
elif os.path.isdir(RBG_MAIN_DIR):
	MAIN_DIR = RBG_MAIN_DIR
elif os.path.isdir(MIT_MAIN_DIR):
	MAIN_DIR = MIT_MAIN_DIR


def read_decode_png(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_png(img, channels=1)
	return img

def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
	return img

def decode_number(label):
	label = tf.expand_dims(label, 0)
	label = tf.strings.to_number(label)
	return label

def map_to_image_label(x, pixel, weighted):
	chest_image = x[0]
	noise_image = x[1]

	y_list = []
	for i in range(2, x.shape[0]):
		y_list.append(x[i])

	if ((weighted == 'True') or (weighted == 'True_bal')):
		y_list.pop()
		sample_weights = x[-1]

	# decode images
	img = read_decode_jpg(chest_image)
	noise_image = read_decode_png(noise_image)

	# get binary noise
	noise_image = tf.image.resize(noise_image, (pixel, pixel))
	noise_image = tf.squeeze(noise_image)
	noise_image = tf.stack([noise_image] * 3, axis = 2)


	# resize
	img = tf.image.resize(img, (pixel, pixel))

	# add noise image
	img = (1 - noise_image) * img + noise_image 

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
	chest_image = x[0]
	noise_image = x[1]

	y_list = []
	for i in range(2, x.shape[0]):
		y_list.append(x[i])

	if ((weighted == 'True') or (weighted == 'True_bal')):
		y_list.pop()
		sample_weights = x[-1]

	# decode images
	img = read_decode_jpg(chest_image)
	noise_image = read_decode_png(noise_image)

	# get binary noise
	noise_image = tf.image.resize(noise_image, (pixel, pixel))
	noise_image = tf.squeeze(noise_image)
	noise_image = tf.stack([noise_image] * 3, axis = 2)


	# resize
	img = tf.image.resize(img, (pixel, pixel))

	# add noise image
	img = (1 - noise_image) * img + noise_image 

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

def sample_y1_on_main(df, y_value, dominant_probability, rng):
	dominant_group = df.index[((df.y0 == y_value) & (df.y1 == y_value))]
	small_group = df.index[((df.y0 == y_value) & (df.y1 == (1 - y_value)))]

	small_probability = 1 - dominant_probability
	# CASE I: Smaller group too large, Dominant group too small
	if len(dominant_group) < (dominant_probability / small_probability) * len(small_group):
		dominant_id = deepcopy(dominant_group).tolist()
		small_id = rng.choice(
			small_group, size=int(
				(small_probability / dominant_probability) * len(dominant_group)
			),
			replace=False).tolist()

	# CASE II: Dominant group too large, smaller group too small
	elif len(small_group) < (small_probability/dominant_probability) * len(dominant_group):
		small_id = deepcopy(small_group).tolist()
		dominant_id = rng.choice(
			dominant_group, size=int(
				(dominant_probability / small_probability) * len(small_group)
			), replace=False).tolist()
	new_ids = small_id + dominant_id
	df_new = df.iloc[new_ids]
	return df_new


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


def get_skewed_data(cand_df, py1d=0.9, py00=0.7, rng=None):
	if rng is None:
		rng = np.random.RandomState(0)
	# --- Fix the conditional distributions 
	cand_df1 = sample_y1_on_main(cand_df, 1, py1d, rng)
	cand_df0 = sample_y1_on_main(cand_df, 0, py1d, rng)
	
	cand_df10 = cand_df1.append(cand_df0)
	cand_df10.reset_index(inplace = True, drop=True)
	
	# --- Fix the marginal 
	final_df = fix_marginal(cand_df10, py00, rng)
	return final_df


def save_created_data(data_frame, experiment_directory, filename):

	txt_df =  f'{MAIN_DIR}/' + data_frame.img_name

	txt_df = txt_df + ',' + data_frame.noise_img

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
	data.columns = ['img_name', 'noise_img'] + \
		[f'y{i}' for i in range(data.shape[1]-2)]

	v_to_drop = [f'y{i}' for i in range(1 + v_dim, data.shape[1]-2)]

	if len(v_to_drop) > 0:
		data.drop(v_to_drop, axis =1, inplace=True)

	txt_df = data.img_name + \
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
	v_mode, alg_step):

	train_data = pd.read_csv(
		f'{experiment_directory}/{v_mode}_train.txt')
	train_data = extract_dim(train_data, v_dim)


	if weighted == 'True':
		train_data = wt.get_permutation_weights(train_data,
			'chexpert', 'tr_consistent')
	elif weighted == 'True_bal':
		train_data = wt.get_permutation_weights(train_data,
			'chexpert', 'bal')

	train_data = train_data.values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]


	validation_data = pd.read_csv(
		f'{experiment_directory}/{v_mode}_valid.txt')

	validation_data = extract_dim(validation_data, v_dim)


	if weighted == 'True':
		validation_data = wt.get_permutation_weights(validation_data,
			'chexpert', 'tr_consistent')
	elif weighted == 'True_bal':
		validation_data = wt.get_permutation_weights(validation_data,
			'chexpert', 'tr_consistent')

	validation_data = validation_data.values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	if alg_step == 'first':
		raise NotImplementedError("not yet")
		first_second_step_idx = pickle.load(
			open(f'{experiment_directory}/first_second_step_idx.pkl', 'rb'))
		train_idx = first_second_step_idx['first']['train_idx']
		valid_idx = first_second_step_idx['first']['valid_idx']

		validation_data = [train_data[i] for i in valid_idx]
		train_data = [train_data[i] for i in train_idx]

	elif alg_step == 'second':
		raise NotImplementedError("not yet")
		first_second_step_idx = pickle.load(
			open(f'{experiment_directory}/first_second_step_idx.pkl', 'rb'))
		train_idx = first_second_step_idx['second']['train_idx']
		train_data = [train_data[i] for i in train_idx]


	test_data_dict = {}
	for dist in [0.1, 0.5, 0.9]:
		test_data = pd.read_csv(
			f'{experiment_directory}/{v_mode}_test_{dist}.txt'
		)
		test_data = extract_dim(test_data, v_dim)
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
			patient_name = df.img_name.iloc[i].split('/')[2]
			study_name = df.img_name.iloc[i].split('/')[3]
			view_name = df.img_name.iloc[i].split('/')[4][:-4]
			pt_noise_image = (f'{experiment_directory}/noise_imgs/'
				f'{group}/{patient_name}_{study_name}_{view_name}.png')
			tf.keras.preprocessing.image.save_img(pt_noise_image,
					tf.reshape(noise_conv, [128, 128, 1]), scale=False
			)
			df.noise_img.iloc[i] = pt_noise_image
	return df


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

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



	label_df['idx'] = label_df.groupby(['y0', 'y1']).cumcount()
	df['idx'] = df.groupby(['y0', 'y1']).cumcount()
	df = df.merge(label_df, on=['idx', 'y0', 'y1'])
	df.drop('idx', axis=1, inplace=True)
	df.reset_index(inplace=True, drop=True)

	if ideal | reverse:
		df = fix_marginal(df, y0_probability=py0, rng=rng)
	df = df[['img_name'] + [f'y{i}' for i in range(df.shape[1] - 1)]]
	df.reset_index(inplace=True, drop=True)
	return df

def create_save_chexpert_lists(experiment_directory, v_mode,
 p_tr=.7, p_val=0.25, random_seed=None):

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	if not os.path.exists(experiment_directory):
		os.makedirs(experiment_directory)


	# --- read in the cleaned image filenames (see cohort_creation)
	if os.path.exists('/data/ddmg/scate/chexpert/'):
		df = pd.read_csv(
			'/data/ddmg/scate/chexpert/penumonia_nofinding_cohort.csv', 
			usecols=['Path', 'y0', 'y1'])
	else:
		raise NotImplementedError('what is the path to penumonia_nofinding_cohort?')
	
	df.rename(columns={'Path':'img_name'}, inplace=True)
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
	# Todo: take v_mode 
	train_valid_df = get_simulated_labels(
		df=train_valid_df,
		py0=None,
		ideal=False,
		reverse=False,
		rng=rng)

	# ---- generate noise images
	train_valid_df = create_noise_patches(experiment_directory, train_valid_df,
	'train_valid', rng)

	train_ids = rng.choice(train_valid_df.shape[0],
		size=int((1.0 - p_val) * train_valid_df.shape[0]), replace=False).tolist()
	train_valid_df['train'] = 0
	train_valid_df.train.loc[train_ids] = 1



	# --- save training data
	train_df = train_valid_df[(train_valid_df.train == 1)].reset_index(drop=True)
	# print(train_df[['y0', 'y1', 'y2']].groupby(['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(train_df.shape[0]), size = int(0.05 * train_df.shape[0]), replace=False).tolist()
	train_df['y0'].iloc[flip_label] = 1.0 - train_df['y0'].iloc[flip_label]

	# --- get the marginal distribution
	fixed_marginal = 1.0 - train_valid_df.y0.mean()

	flip_label = rng.choice(range(train_df.shape[0]), size = int(0.05 * train_df.shape[0]), replace=False).tolist()
	train_df['y1'].iloc[flip_label] = 1.0 - train_df['y1'].iloc[flip_label]

	flip_label = rng.choice(range(train_df.shape[0]), size = int(0.05 * train_df.shape[0]), replace=False).tolist()
	train_df['y2'].iloc[flip_label] = 1.0 - train_df['y2'].iloc[flip_label]

	print("===== training data ======")
	print(train_df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
		["mean", "count"]).reset_index())

	print(train_df[[f'y{i}' for i in range(13)]].groupby(
	[f'y{i}' for i in range(1, 13)]).agg(
	    ["mean", "count"]).reset_index())

	print(train_df.y0.value_counts(normalize=True))

	save_created_data(train_df, experiment_directory=experiment_directory,
		filename=f'{v_mode}_train')

	# --- save validation data
	valid_df = train_valid_df[(train_valid_df.train == 0)].reset_index(drop=True)
	# print(valid_df[['y0', 'y1', 'y2']].groupby(['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(valid_df.shape[0]), size = int(0.05 * valid_df.shape[0]), replace=False).tolist()
	valid_df['y0'].iloc[flip_label] = 1.0 - valid_df['y0'].iloc[flip_label]

	flip_label = rng.choice(range(valid_df.shape[0]), size = int(0.05 * valid_df.shape[0]), replace=False).tolist()
	valid_df['y1'].iloc[flip_label] = 1.0 - valid_df['y1'].iloc[flip_label]

	flip_label = rng.choice(range(valid_df.shape[0]), size = int(0.05 * valid_df.shape[0]), replace=False).tolist()
	valid_df['y2'].iloc[flip_label] = 1.0 - valid_df['y2'].iloc[flip_label]

	print("===== valid data ======")
	print(valid_df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
		["mean", "count"]).reset_index())

	print(valid_df[[f'y{i}' for i in range(13)]].groupby(
	[f'y{i}' for i in range(1, 13)]).agg(
	    ["mean", "count"]).reset_index())

	print(valid_df.y0.value_counts(normalize=True))


	save_created_data(valid_df, experiment_directory=experiment_directory,
		filename=f'{v_mode}_valid')

	# ------------------------------------------- #
	# -------------- get the test data ---------- #
	# ------------------------------------------- #

	test_df = df[(df.train_valid_ids == 0)].reset_index(drop=True)
	test_df.drop(['train_valid_ids'], axis=1, inplace=True)

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

	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y0'].iloc[flip_label] = 1.0 - curr_test_df['y0'].iloc[flip_label]

	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y1'].iloc[flip_label] = 1.0 - curr_test_df['y1'].iloc[flip_label]

	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y2'].iloc[flip_label] = 1.0 - curr_test_df['y2'].iloc[flip_label]

	print("===== test 0.9 data ======")
	print(curr_test_df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
		["mean", "count"]).reset_index())

	print(curr_test_df[[f'y{i}' for i in range(13)]].groupby(
	[f'y{i}' for i in range(1, 13)]).agg(
	    ["mean", "count"]).reset_index())

	print(curr_test_df.y0.value_counts(normalize=True))

	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename=f'{v_mode}_test_0.9')

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
	# print(curr_test_df[['y0', 'y1', 'y2']].groupby(
	# 	['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y0'].iloc[flip_label] = 1.0 - curr_test_df['y0'].iloc[flip_label]

	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y1'].iloc[flip_label] = 1.0 - curr_test_df['y1'].iloc[flip_label]

	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y2'].iloc[flip_label] = 1.0 - curr_test_df['y2'].iloc[flip_label]

	print("===== test 0.5 data ======")
	print(curr_test_df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
		["mean", "count"]).reset_index())

	print(curr_test_df[[f'y{i}' for i in range(13)]].groupby(
	[f'y{i}' for i in range(1, 13)]).agg(
	    ["mean", "count"]).reset_index())

	print(curr_test_df.y0.value_counts(normalize=True))

	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename=f'{v_mode}_test_0.5')

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
	# print(curr_test_df[['y0', 'y1', 'y2']].groupby(
	# 	['y0', 'y1', 'y2']).size().reset_index())
	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y0'].iloc[flip_label] = 1.0 - curr_test_df['y0'].iloc[flip_label]

	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y1'].iloc[flip_label] = 1.0 - curr_test_df['y1'].iloc[flip_label]

	flip_label = rng.choice(range(curr_test_df.shape[0]), size = int(0.05 * curr_test_df.shape[0]), replace=False).tolist()
	curr_test_df['y2'].iloc[flip_label] = 1.0 - curr_test_df['y2'].iloc[flip_label]

	print("===== test 0.1 data ======")
	print(curr_test_df[['y1', 'y2', 'y0']].groupby(['y1', 'y2']).agg(
		["mean", "count"]).reset_index())

	print(curr_test_df[[f'y{i}' for i in range(13)]].groupby(
	[f'y{i}' for i in range(1, 13)]).agg(
	    ["mean", "count"]).reset_index())

	print(curr_test_df.y0.value_counts(normalize=True))

	save_created_data(curr_test_df, experiment_directory=experiment_directory,
		filename=f'{v_mode}_test_0.1')


def build_input_fns(chexpert_data_dir, v_mode,
	weighted='False', p_tr=.7, p_val=0.25, v_dim=0, random_seed=None,
	alg_step='None'):
	experiment_directory = f'{chexpert_data_dir}/experiment_data/rs{random_seed}'

	# --- generate splits if they dont exist
	if not os.path.exists(
		f'{experiment_directory}/{v_mode}_train.txt'):

		create_save_chexpert_lists(
			experiment_directory=experiment_directory,
			v_mode=v_mode, 
			p_tr=p_tr,
			p_val=p_val,
			random_seed=random_seed)

	# --load splits

	train_data, valid_data, shifted_data_dict = load_created_data(
		experiment_directory=experiment_directory, weighted=weighted,
		v_dim=v_dim, v_mode=v_mode, alg_step=alg_step)

	# --this helps auto-set training steps at train time
	train_data_size = len(train_data)

	# Build an iterator over training batches.
	def train_input_fn(params):
		map_to_image_label_wrapper = functools.partial(map_to_image_label,
			pixel=params['pixel'], weighted=params['weighted'])
		batch_size = params['batch_size']
		num_epochs = params['num_epochs']

		dataset = tf.data.Dataset.from_tensor_slices(train_data)
		dataset = dataset.map(map_to_image_label_wrapper, num_parallel_calls=1)
		# dataset = dataset.shuffle(int(train_data_size * 0.05)).batch(batch_size
		# ).repeat(num_epochs)
		dataset = dataset.batch(batch_size).repeat(num_epochs)
		return dataset

	# Build an iterator over validation batches

	def valid_input_fn(params):
		map_to_image_label_wrapper = functools.partial(map_to_image_label,
			pixel=params['pixel'], weighted=params['weighted'])
		batch_size = params['batch_size']
		valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
		valid_dataset = valid_dataset.map(map_to_image_label_wrapper,
			num_parallel_calls=1)
		valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(
			1)
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

	return train_data_size, train_input_fn, valid_input_fn, final_valid_input_fn, eval_input_fn_creater
