"""Functions to create the chexpert datasets."""
import os, shutil
import functools
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf

from chexpert_support_device import weighting as wt

SOTO_MAIN_DIR = '/nfs/turbo/coe-soto'
RBG_MAIN_DIR = '/nfs/turbo/coe-rbg'
MIT_MAIN_DIR = '/data/ddmg/slabs'

if os.path.isdir(SOTO_MAIN_DIR):
	MAIN_DIR = SOTO_MAIN_DIR
elif os.path.isdir(RBG_MAIN_DIR):
	MAIN_DIR = RBG_MAIN_DIR
elif os.path.isdir(MIT_MAIN_DIR):
	MAIN_DIR = MIT_MAIN_DIR


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
	y0 = x[1]
	y1 = x[2]
	y2 = x[3]

	if weighted == 'True':
		sample_weights = x[4]

	# decode images
	img = read_decode_jpg(chest_image)

	# resize, rescale  image
	img = tf.image.resize(img, (pixel, pixel))
	img = img / 255

	# get the label vector
	y0 = decode_number(y0)
	y1 = decode_number(y1)
	y2 = decode_number(y2)
	labels = tf.concat([y0, y1, y2], axis=0)

	if weighted == 'True':
		sample_weights = decode_number(sample_weights)
	else:
		sample_weights = None

	labels_and_weights = {'labels': labels, 'sample_weights': sample_weights}
	return img, labels_and_weights

def map_to_image_label_test(x, pixel, weighted):
	""" same as normal function by makes sure to not use
	weighting. """
	chest_image = x[0]
	y0 = x[1]
	y1 = x[2]
	y2 = x[3]

	# decode images
	img = read_decode_jpg(chest_image)

	# resize, rescale  image
	img = tf.image.resize(img, (pixel, pixel))
	img = img / 255

	# get the label vector
	y0 = decode_number(y0)
	y1 = decode_number(y1)
	y2 = decode_number(y2)
	labels = tf.concat([y0, y1, y2], axis=0)

	if weighted == 'True':
		sample_weights = tf.ones_like(y0)
	else:
		sample_weights = None

	labels_and_weights = {'labels': labels, 'sample_weights': sample_weights}
	return img, labels_and_weights


def sample_y2_on_y1(df, y0_value, y1_value, dominant_probability, rng):
	dominant_group = df.index[
		((df.y0 == y0_value) & (df.y1 == y1_value) & (df.y2 == y1_value))
	]
	small_group = df.index[
		((df.y0 == y0_value) & (df.y1 == y1_value) & (df.y2 == (1 - y1_value)))
	]

	small_probability = 1 - dominant_probability

	# CASE I: Smaller group too large, Dominant group too small
	# If the dominant group is smaller than dominant probability*len(group)
	# Truncate the size of the small group based on the dominant probability
	if len(dominant_group) < (dominant_probability / small_probability) * len(small_group):
		dominant_id = deepcopy(dominant_group).tolist()
		small_id = rng.choice(
			small_group, size=int(
				(small_probability / dominant_probability) * len(dominant_group)
			),
			replace=False).tolist()

	# CASE II: Dominant group too large, smaller group too small
	# If the small group if smaller than small probability*len(group)
	# Truncate the size of the large group based on the small probability
	elif len(small_group) < (small_probability / dominant_probability) * len(dominant_group):
		small_id = deepcopy(small_group).tolist()
		dominant_id = rng.choice(
			dominant_group, size=int(
				(dominant_probability / small_probability) * len(small_group)
			), replace=False).tolist()
	new_ids = small_id + dominant_id
	df_new = df.iloc[new_ids]
	return df_new


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


def get_skewed_data(cand_df, py1d=0.9, py2d=0.9, py00=0.7, rng=None):
	if rng is None:
		rng = np.random.RandomState(0)
	# --- Fix the conditional distributions of y2
	cand_df11 = sample_y2_on_y1(cand_df, 1, 1, py2d, rng)
	cand_df10 = sample_y2_on_y1(cand_df, 1, 0, py2d, rng)
	cand_df01 = sample_y2_on_y1(cand_df, 0, 1, py2d, rng)
	cand_df00 = sample_y2_on_y1(cand_df, 0, 0, py2d, rng)

	new_cand_df = cand_df11.append(cand_df10).append(cand_df01).append(cand_df00)
	new_cand_df.reset_index(inplace=True, drop=True)

	# --- Fix the conditional distributions of y1
	cand_df1 = sample_y1_on_main(new_cand_df, 1, py1d, rng)
	cand_df0 = sample_y1_on_main(new_cand_df, 0, py1d, rng)

	cand_df10 = cand_df1.append(cand_df0)
	cand_df10.reset_index(inplace=True, drop=True)

	# --- Fix the marginal
	final_df = fix_marginal(cand_df10, py00, rng)

	return final_df

def save_created_data(data_frame, experiment_directory, filename):
	D = data_frame.shape[1]

	if 'Path' in data_frame.columns:
		txt_df = f'{MAIN_DIR}/' + data_frame.Path
	else: 
		txt_df =  data_frame.file_name
	for i in range(D-1):
		txt_df = txt_df + ',' + data_frame[f'y{i}'].astype(str)

	txt_df.to_csv(f'{experiment_directory}/{filename}.txt',
		index=False)


def load_created_data(chexpert_data_dir, random_seed, v_mode, v_dim, skew_train, weighted):
	experiment_directory = f'{chexpert_data_dir}/experiment_data/rs{random_seed}'

	skew_str = 'skew' if skew_train == 'True' else 'unskew'

	if v_mode == 'noisy': 
		v_str = f'noisy{v_dim}_'
	elif v_mode == 'corry':
		v_str = f'corry{v_dim}_'
	else: 
		v_str = ''

	train_data = pd.read_csv(
		f'{experiment_directory}/{v_str}{skew_str}_train.txt')

	if weighted == 'True':
		train_data = wt.get_simple_weights(train_data)

	train_data = train_data.values.tolist()
	train_data = [
		tuple(train_data[i][0].split(',')) for i in range(len(train_data))
	]

	validation_data = pd.read_csv(
		f'{experiment_directory}/{v_str}{skew_str}_valid.txt')

	if weighted == 'True':
		validation_data = wt.get_simple_weights(validation_data)

	validation_data = validation_data.values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

	varying_joint_test_data_dict = {}
	for pskew in pskew_list:
		test_data = pd.read_csv(
			f'{experiment_directory}/{v_str}{pskew}_test.txt'
		).values.tolist()
		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		varying_joint_test_data_dict[pskew] = test_data


	fixed_joint_skew_test_data_dict = {}
	for pskew in pskew_list:
		test_data = pd.read_csv(
			f'{experiment_directory}/{v_str}{pskew}_fj09_test.txt'
		).values.tolist()
		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		fixed_joint_skew_test_data_dict[pskew] = test_data


	fixed_joint_unskew_test_data_dict = {}
	for pskew in pskew_list:
		test_data = pd.read_csv(
			f'{experiment_directory}/{v_str}{pskew}_fj05_test.txt'
		).values.tolist()
		test_data = [
			tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]
		fixed_joint_unskew_test_data_dict[pskew] = test_data


	test_data_dict = {
		'varying_joint' : varying_joint_test_data_dict,
		'fixed_joint_0.9' : fixed_joint_skew_test_data_dict,
		'fixed_joint_0.5': fixed_joint_unskew_test_data_dict
	}
	return train_data, validation_data, test_data_dict

def get_noisy_data(data, v_dim):
	data = data['0'].str.split(",", expand=True)
	N, D = data.shape 
	data.columns = ['file_name'] +  [f'y{i}' for i in range(D - 1)]

	for i in range(D-1): 
		data[f'y{i}'] = data[f'y{i}'].astype(np.float32)

	for i in range(D-1, D+v_dim-1):
		data[f'y{i}'] = np.random.binomial(
			n=1, p=0.5, size=(N, 1)).astype(np.float32)
	return data
	
def create_additional_v(experiment_directory, v_mode, v_dim,
	skew_train, weighted):

	skew_str = 'skew' if skew_train == 'True' else 'unskew'

	# -- Get noisy training data
	train_data = pd.read_csv(
		f'{experiment_directory}/{skew_str}_train.txt')

	train_data = get_noisy_data(train_data, v_dim)
	save_created_data(train_data, 
		experiment_directory=experiment_directory, 
		filename=f'noisy{v_dim}_skew_train')

	# -- Get noisy valid data 
	valid_data = pd.read_csv(
		f'{experiment_directory}/{skew_str}_valid.txt')

	valid_data = get_noisy_data(valid_data, v_dim)
	save_created_data(valid_data, 
		experiment_directory=experiment_directory, 
		filename=f'noisy{v_dim}_skew_valid')

	# -- get noisy test data
	pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

	for pskew in pskew_list:
		# --- get the non-fixed joint data
		test_data = pd.read_csv(
			f'{experiment_directory}/{pskew}_test.txt')

		test_data = get_noisy_data(test_data, v_dim)
		save_created_data(test_data, 
			experiment_directory=experiment_directory, 
			filename=f'noisy{v_dim}_{pskew}_test')

		# --- fixed joint 0.9 
		test_data = pd.read_csv(
			f'{experiment_directory}/{pskew}_fj09_test.txt'
		)

		test_data = get_noisy_data(test_data, v_dim)
		save_created_data(test_data, 
			experiment_directory=experiment_directory, 
			filename=f'noisy{v_dim}_{pskew}_fj09_test')

		# --- fixed joint 0.5 
		test_data = pd.read_csv(
			f'{experiment_directory}/{pskew}_fj05_test.txt'
		)

		test_data = get_noisy_data(test_data, v_dim)
		save_created_data(test_data, 
			experiment_directory=experiment_directory, 
			filename=f'noisy{v_dim}_{pskew}_fj05_test')


def create_save_chexpert_lists(chexpert_data_dir, p_tr=.7, p_val=0.25,
	random_seed=None):
	p_dom = 0.9
	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	experiment_directory = f'{chexpert_data_dir}/experiment_data/rs{random_seed}'

	if not os.path.exists(experiment_directory):
		os.makedirs(experiment_directory)


	# --- read in the cleaned image filenames (see cohort_creation)
	df = pd.read_csv(f'{chexpert_data_dir}/penumonia_nofinding_sd_cohort.csv')

	# ---- split into train and test patients
	tr_val_candidates = rng.choice(df.patient.unique(),
		size=int(len(df.patient.unique()) * p_tr), replace=False).tolist()
	ts_candidates = list(set(df.patient.unique()) - set(tr_val_candidates))

	# --- split training into training and validation
	tr_candidates = rng.choice(tr_val_candidates,
		size=int((1 - p_val) * len(tr_val_candidates)), replace=False).tolist()
	val_candidates = list(set(tr_val_candidates) - set(tr_candidates))

	tr_candidates_df = df[(df.patient.isin(tr_candidates))].reset_index(drop=True)
	val_candidates_df = df[(df.patient.isin(val_candidates))].reset_index(
		drop=True)
	ts_candidates_df = df[(df.patient.isin(ts_candidates))].reset_index(drop=True)

	# --- checks
	assert len(ts_candidates) + len(tr_candidates) + len(val_candidates) == len(
		df.patient.unique())
	assert len(set(ts_candidates) & set(tr_candidates)) == 0
	assert len(set(ts_candidates) & set(val_candidates)) == 0
	assert len(set(tr_candidates) & set(val_candidates)) == 0

	# --- get train datasets
	tr_sk_df = get_skewed_data(tr_candidates_df, py1d=p_dom, py2d=p_dom, py00=0.7,
		rng=rng)
	tr_sk_df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
	save_created_data(tr_sk_df, experiment_directory=experiment_directory,
		filename='skew_train')

	tr_usk_df = get_skewed_data(tr_candidates_df, py1d=0.5, py2d=0.5, py00=0.7,
		rng=rng)
	tr_usk_df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
	save_created_data(tr_usk_df, experiment_directory=experiment_directory,
		filename='unskew_train')

	# --- get validation datasets
	val_sk_df = get_skewed_data(val_candidates_df, py1d=p_dom, py2d=p_dom, py00=0.7,
		rng=rng)
	val_sk_df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
	save_created_data(val_sk_df, experiment_directory=experiment_directory,
		filename='skew_valid')

	val_usk_df = get_skewed_data(val_candidates_df, py1d=0.5, py2d=0.5, py00=0.7,
		rng=rng)
	val_usk_df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
	save_created_data(val_usk_df, experiment_directory=experiment_directory,
		filename='unskew_valid')

	# --- get test
	pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
	for pskew in pskew_list:
		ts_sk_df = get_skewed_data(ts_candidates_df, py1d=pskew, py2d=pskew, py00=0.7,
			rng=rng)
		ts_sk_df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
		save_created_data(ts_sk_df, experiment_directory=experiment_directory,
			filename=f'{pskew}_test')

	# get test fixed aux joint skewed
	for pskew in pskew_list:
		ts_sk_df = get_skewed_data(ts_candidates_df, py1d=pskew, py2d=p_dom, py00=0.7,
			rng=rng)
		ts_sk_df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
		save_created_data(ts_sk_df, experiment_directory=experiment_directory,
			filename=f'{pskew}_fj09_test')

	# get test fixed aux joint
	for pskew in pskew_list:
		ts_sk_df = get_skewed_data(ts_candidates_df, py1d=pskew, py2d=0.5, py00=0.7,
			rng=rng)
		ts_sk_df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
		save_created_data(ts_sk_df, experiment_directory=experiment_directory,
			filename=f'{pskew}_fj05_test')


def build_input_fns(chexpert_data_dir, v_mode, skew_train='False',
	weighted='False', p_tr=.7, p_val=0.25, v_dim=0, random_seed=None):
	experiment_directory = f'{chexpert_data_dir}/experiment_data/rs{random_seed}'

	# --- generate splits if they dont exist
	if not os.path.exists(
		f'{experiment_directory}/skew_train.txt'):

		create_save_chexpert_lists(
			chexpert_data_dir=chexpert_data_dir,
			p_tr=p_tr,
			p_val=p_val,
			random_seed=random_seed)

	if v_mode == 'noisy':
		if not os.path.exists(
			f'{experiment_directory}/noisy_skew_train.txt'):
			create_additional_v(experiment_directory=experiment_directory, 
				v_mode=v_mode, v_dim=v_dim,
				skew_train=skew_train, weighted=weighted)

	if v_mode == 'corry':
		if not os.path.exists(
			f'{experiment_directory}/corry_skew_train.txt'):
			raise NotImplementedError("not implemented yet")

	# --load splits
	train_data, valid_data, shifted_data_dict = load_created_data(
		chexpert_data_dir=chexpert_data_dir, random_seed=random_seed,
		v_mode=v_mode, v_dim=v_dim, skew_train=skew_train,
		weighted=weighted)

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

	# Build an iterator over the heldout set (shifted distribution).
	def eval_input_fn_creater(py, params, fixed_joint=False, aux_joint_skew=0.5):
		map_to_image_label_wrapper = functools.partial(map_to_image_label_test,
			pixel=params['pixel'], weighted=params['weighted'])
		if fixed_joint:
			if aux_joint_skew == 0.9:
				shifted_test_data = shifted_data_dict['fixed_joint_0.9'][py]
			elif aux_joint_skew == 0.5:
				shifted_test_data = shifted_data_dict['fixed_joint_0.5'][py]
		else:
			shifted_test_data = shifted_data_dict['varying_joint'][py]
		batch_size = params['batch_size']

		def eval_input_fn():
			eval_shift_dataset = tf.data.Dataset.from_tensor_slices(shifted_test_data)
			eval_shift_dataset = eval_shift_dataset.map(map_to_image_label_wrapper)
			eval_shift_dataset = eval_shift_dataset.batch(batch_size).repeat(1)
			return eval_shift_dataset
		return eval_input_fn

	return train_data_size, train_input_fn, valid_input_fn, eval_input_fn_creater
