"""Creates data for support device+sex shortcut Chexpert."""
import os
import argparse
import pandas as pd
import numpy as np

from copy import deepcopy
pd.set_option('mode.chained_assignment', None)


def create_cohort(save_directory):
	""" Function that creates a csv with the cohort"""
	# -- import the data
	trdf = pd.read_csv('/nfs/turbo/coe-rbg/CheXpert-v1.0/train.csv')
	vdf = pd.read_csv('/nfs/turbo/coe-rbg/CheXpert-v1.0/valid.csv')
	df = trdf.append(vdf)
	del trdf, vdf

	# -- keep only healthy, pneumonia and support device
	df = df[(
		(df['No Finding'] == 1) | (df['Pneumonia'] == 1) | (df['Support Devices'] == 1)
	)]

	# print(df.Pneumonia.value_counts(dropna=False, normalize=True))
	# print(df['Support Devices'].value_counts(dropna=False))

	# -- clean up a bit
	df['patient'] = df.Path.str.extract(r'(patient)(\d+)')[1]
	df['study'] = df.Path.str.extract(r'(study)(\d+)')[1].astype(int)
	df['uid'] = df['patient'] + "_" + df['study'].astype(str)
	df = df[
		['uid', 'patient', 'study', 'Sex', 'Frontal/Lateral',
		'Pneumonia', 'Support Devices', 'Path']
	]

	# current_uid = len(df.uid.unique())
	# print(f'Total uids {current_uid}')

	# get the main outcome
	df['y0'] = df['Pneumonia'].copy()
	df.y0.fillna(0, inplace=True)
	df.y0[(df.y0 == -1)] = 1

	# print(df.y0.value_counts(dropna=False, normalize=True))

	# get the first auxiliary label
	df = df[(df.Sex != 'Unknown')]
	df['y1'] = (df.Sex == 'Male').astype(int)
	df.drop('Sex', axis=1, inplace=True)

	# print(f'Lost {100*(current_uid - len(df.uid.unique()))/current_uid:.3f}% because of unknown sex')
	# current_uid = len(df.uid.unique())

	# get the second auxiliary label
	df['y2'] = df['Support Devices'].copy()
	df.y2.fillna(0, inplace=True)
	df.y2[(df.y2 == -1)] = 1
	# print(df.y2.value_counts(dropna = False, normalize = True))

	# keep only studies with frontal views
	df['frontal'] = (df['Frontal/Lateral'] == 'Frontal').astype(int)
	df = df[(df.frontal == 1)]

	# more clean ups
	# print(f'Lost {100*(current_uid - len(df.uid.unique()))/current_uid:.3f}% because they dont have frontal views')
	# current_uid = len(df.uid.unique())

	df.drop_duplicates(subset=['uid'], inplace=True)
	# print(f'Lost {100*(current_uid - df.shape[0])/current_uid:.3f}% because they have duplicates')
	# current_uid = len(df.uid.unique())

	df.drop(
		['Frontal/Lateral', 'frontal', 'Pneumonia', 'Support Devices'],
		axis=1, inplace=True)
	# print(df.head())

	# save
	df.to_csv(f'{save_directory}/penumonia_nofinding_sd_cohort.csv', index=False)


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


def save_created_data(df, experiment_directory, filename):
	IMG_MAIN_DIR = "/nfs/turbo/coe-rbg/"
	df['Path'] = IMG_MAIN_DIR + df['Path']
	df.drop(['uid', 'patient', 'study'], axis=1, inplace=True)
	df.to_csv(f'{experiment_directory}/{filename}.csv', index=False)


def create_save_chexpert_lists(save_directory, p_tr=.7, p_val=0.25,
	random_seed=None):

	if random_seed is None:
		rng = np.random.RandomState(0)
		experiment_directory = f'{save_directory}/experiment_data/rsNone'
	else:
		rng = np.random.RandomState(random_seed)
		experiment_directory = f'{save_directory}/experiment_data/rs{random_seed}'

	if not os.path.exists(experiment_directory):
		os.makedirs(experiment_directory)

	# --- read in the cleaned image filenames (see chexpert_creation)
	df = pd.read_csv(f'{save_directory}/penumonia_nofinding_sd_cohort.csv')

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
	tr_sk_df = get_skewed_data(tr_candidates_df, py1d=0.9, py2d=0.9, py00=0.7,
		rng=rng)
	save_created_data(tr_sk_df, experiment_directory=experiment_directory,
		filename='skew_train')

	tr_usk_df = get_skewed_data(tr_candidates_df, py1d=0.5, py2d=0.5, py00=0.7,
		rng=rng)
	save_created_data(tr_usk_df, experiment_directory=experiment_directory,
		filename='unskew_train')

	# --- get validation datasets
	val_sk_df = get_skewed_data(val_candidates_df, py1d=0.9, py2d=0.9, py00=0.7,
		rng=rng)
	save_created_data(val_sk_df, experiment_directory=experiment_directory,
		filename='skew_valid')

	val_usk_df = get_skewed_data(val_candidates_df, py1d=0.5, py2d=0.5, py00=0.7,
		rng=rng)
	save_created_data(val_usk_df, experiment_directory=experiment_directory,
		filename='unskew_valid')

	# --- get test
	pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
	for pskew in pskew_list:
		ts_sk_df = get_skewed_data(ts_candidates_df, py1d=pskew, py2d=pskew, py00=0.7,
			rng=rng)
		save_created_data(ts_sk_df, experiment_directory=experiment_directory,
			filename=f'{pskew}_test')


def main(save_directory, random_seed, p_tr=0.7, p_val=0.25):
	# --- create the main csv file with the full cohort
	create_cohort(save_directory)

	# --- create the training/validation/testing splits
	create_save_chexpert_lists(save_directory, p_tr=p_tr, p_val=p_val,
	random_seed=random_seed)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--save_directory', '-save_directory',
		help="Directory where the final cohort will be saved",
		type=str)

	parser.add_argument('--random_seed', '-random_seed',
		help="Random seed for data split",
		type=int)

	parser.add_argument('--p_tr', '-p_tr',
		help="Proportion to use for the training",
		default=0.7,
		type=float)

	parser.add_argument('--p_val', '-p_val',
		help="Proportion to use for validation",
		default=0.25,
		type=float)

	args = vars(parser.parse_args())
	main(**args)
