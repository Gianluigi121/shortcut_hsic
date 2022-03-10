import os, shutil
import functools
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf

MAIN_DIR = '/nfs/turbo/coe-rbg/zhengji/age/'

"""Dataset"""
# y0_probability: The probability of sex label=0
def fix_marginal(df, y0_probability, rng):
	y0_group = df.index[(df.y0 == 0)]
	y1_group = df.index[(df.y0 == 1)]
	
	y1_probability = 1 - y0_probability 

	if len(y0_group) < (y0_probability/y1_probability) * len(y1_group):
		y0_ids = deepcopy(y0_group).tolist()
		y1_ids = rng.choice(
			y1_group, size = int((y1_probability/y0_probability) * len(y0_group)),
			replace = False).tolist()
   
	elif len(y1_group) < (y1_probability/y0_probability) * len(y0_group):
		y1_ids = deepcopy(y1_group).tolist()
		y0_ids = rng.choice(
			y0_group, size = int( (y0_probability/y1_probability)*len(y1_group)), 
			replace = False
		).tolist()
  
	dff = df.iloc[y1_ids + y0_ids]
	dff.reset_index(inplace = True, drop=True)
	reshuffled_ids = rng.choice(dff.index, size = len(dff.index), replace=False).tolist()
	dff = dff.iloc[reshuffled_ids].reset_index(drop = True)
	return dff

# p_sex: probability of sex=1
def get_skewed_data(df, p_sex=0.9, rng=None):
	if rng is None:
		rng = np.random.RandomState(0)
	final_df = fix_marginal(df, p_sex, rng)
	return final_df

def save_created_data(df, experiment_directory, filename):
	df.to_csv(f'{experiment_directory}/{filename}.csv',
		index=False)

# p_train: Probability of the training and validation dataset
def create_save_chexpert_lists(experiment_directory, p_train=0.9, random_seed=None):
	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	if not os.path.isdir(experiment_directory):
		os.mkdir(experiment_directory)

	# --- read in the cleaned image filenames (see chexpert_creation)
	df = pd.read_csv(MAIN_DIR+'penumonia_nofinding_cohort.csv')

	# ---- split into train and test patients
	tr_val_candidates = rng.choice(df.patient.unique(),
		size = int(len(df.patient.unique())*p_train), replace = False).tolist()
	ts_candidates = list(set(df.patient.unique()) - set(tr_val_candidates))

	# --- split training into training and validation
	# TODO: don't hard code the validation percent
	tr_candidates = rng.choice(tr_val_candidates,
		size=int(0.8 * len(tr_val_candidates)), replace=False).tolist()
	val_candidates = list(set(tr_val_candidates) - set(tr_candidates))

	tr_candidates_df = df[(df.patient.isin(tr_candidates))].reset_index(drop=True)
	val_candidates_df = df[(df.patient.isin(val_candidates))].reset_index(drop=True)
	ts_candidates_df = df[(df.patient.isin(ts_candidates))].reset_index(drop=True)

	# --- checks
	assert len(ts_candidates) + len(tr_candidates) + len(val_candidates) == len(df.patient.unique())
	assert len(set(ts_candidates) & set(tr_candidates)) == 0
	assert len(set(ts_candidates) & set(val_candidates)) == 0
	assert len(set(tr_candidates) & set(val_candidates)) == 0

	# --- get train datasets
	tr_sk_df = get_skewed_data(tr_candidates_df, p_sex=0.9, rng=rng)
	save_created_data(tr_sk_df, experiment_directory=experiment_directory,
        filename='skew_train')

	tr_usk_df = get_skewed_data(tr_candidates_df, p_sex=0.5, rng=rng)
	save_created_data(tr_usk_df, experiment_directory=experiment_directory,
        filename='unskew_train')

    # --- get validation datasets
	val_sk_df = get_skewed_data(val_candidates_df, p_sex=0.9, rng=rng)
	save_created_data(val_sk_df, experiment_directory=experiment_directory,
        filename='skew_valid')

	val_usk_df = get_skewed_data(val_candidates_df, p_sex=0.5, rng=rng)
	save_created_data(val_usk_df, experiment_directory=experiment_directory,
        filename='unskew_valid')

	prob_list = [0.5, 0.9]
	for skew_prob in prob_list:
		ts_sk_df = get_skewed_data(ts_candidates_df, p_sex=skew_prob, rng=rng)
		save_created_data(ts_sk_df, experiment_directory=experiment_directory,
			filename=f'skew_test_{skew_prob}')

def load_ds_from_csv(experiment_directory, skew_train, params):
	skew_str = 'skew' if skew_train == 'True' else 'unskew'

	train_csv_dir = f'{experiment_directory}/{skew_str}_train.csv'
	train_ds = create_dataset(train_csv_dir, params)

	valid_csv_dir = f'{experiment_directory}/{skew_str}_valid.csv'
	valid_ds = create_dataset(valid_csv_dir, params)

	test_ds_dict = {}
	for pskew in [0.5, 0.9]:
		test_csv_dir = f'{experiment_directory}/skew_test_{pskew}.csv'
		test_ds_dict[pskew] = create_dataset(test_csv_dir, params)

	return train_ds, valid_ds, test_ds_dict

def map_to_image_label(img_dir, label):
    img = tf.io.read_file(img_dir)
    img = tf.image.decode_jpeg(img, channels=3)

    # Resize and rescale the image
    img_height = 128
    img_width = 128
    
    img = tf.image.resize(img, (img_height, img_width))
    img = img / 255

    return img, label

def create_dataset(csv_dir, params):
    df = pd.read_csv(csv_dir)
    df.Path = '/nfs/turbo/coe-rbg/'+df.Path
    file_paths = df['Path'].values
    label = df['Age'].values   
    label = tf.cast(label, tf.float32)

    batch_size = params['batch_size']
    ds = tf.data.Dataset.from_tensor_slices((file_paths, label))
    ds = ds.map(map_to_image_label).batch(batch_size)
    return ds

# create_save_chexpert_lists(MAIN_DIR + 'data')
# params = {'batch_size': 32}
# train_ds, valid_ds, test_ds_dict = load_ds_from_csv(MAIN_DIR+'data', True, params)

# for x, y in train_ds.take(2):
# 	print(y)