""" Script to get the predictions for the cross validated model."""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
import copy
from pathlib import Path
from random import sample
import functools
from sklearn.metrics import roc_auc_score
import multiprocessing
import tqdm
import matplotlib.pyplot as plt
import itertools


import tensorflow as tf
import chexpert_support_device.data_builder as db
from chexpert_support_device import configurator
import shared.train_utils as utils
from pathlib import Path


def get_test_data(config, pskew, base_dir, fixed_joint, aux_joint_skew):
		"""Function to get the data."""
		experiment_directory = (
			f"{base_dir}/experiment_data/rs{config['random_seed']}")

		if fixed_joint:
			if aux_joint_skew == 0.9:
				test_data = pd.read_csv(
					f'{experiment_directory}/{pskew}_fj09_test.txt'
				).values.tolist()
			else:
				test_data = pd.read_csv(
					f'{experiment_directory}/{pskew}_fj05_test.txt'
				).values.tolist()
		else:
			test_data = pd.read_csv(
				f'{experiment_directory}/{pskew}_test.txt'
				).values.tolist()

		test_data = [
				tuple(test_data[i][0].split(',')) for i in range(len(test_data))
		]

		map_to_image_label_given_pixel = functools.partial(db.map_to_image_label_test,
				pixel=config['pixel'], weighted=config['weighted'])
		test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
		test_dataset = test_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
		test_dataset = test_dataset.batch(config['batch_size'],
			drop_remainder=False).repeat(1)
		return test_dataset


def get_last_saved_model(estimator_dir):
		""" Function to get the last saved model"""
		subdirs = [x for x in Path(estimator_dir).iterdir()
				if x.is_dir() and 'temp' not in str(x)]
		try:
				latest_model_dir = str(sorted(subdirs)[-1])
				loaded = tf.saved_model.load(latest_model_dir)
				model = loaded.signatures["serving_default"]
		except:
				print(estimator_dir)
		return model

def get_pred(pskew, random_seed, fixed_joint, aux_joint_skew,
 hash_string, base_dir):
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')
	model = get_last_saved_model(hash_dir)

	config_dir = os.path.join(base_dir, 'tuning', hash_string, 'config.pkl')
	config = pickle.load(open(config_dir, 'rb'))

	test_dataset = get_test_data(
		config=config,
		pskew=pskew,
		base_dir=base_dir,
		fixed_joint=fixed_joint,
		aux_joint_skew=aux_joint_skew
	)

	pred_df_list = []
	for batch_id, examples in enumerate(test_dataset):
			print(f'{batch_id}')
			x, labels_weights = examples
			predictions = model(tf.convert_to_tensor(x))['probabilities']

			pred_df = pd.DataFrame(labels_weights['labels'].numpy())
			pred_df.columns = ['y'] + [f'v{i}' for i in range(pred_df.shape[1] -1)]

			pred_df['predictions'] = predictions.numpy()
			pred_df['pred_class'] = (pred_df.predictions >= 0.5)*1.0

			pred_df_list.append(pred_df)

	pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True)
	pred_df['pskew'] = pskew
	return pred_df

def get_optimal_pred_for_random_seed(random_seed, pixel, batch_size,
	fixed_joint, aux_joint_skew, model_name, xv_mode, experiment_name,
	base_dir, **args):

	# -- get the optimal model configs
	optimal_configs = pd.read_csv(
			(f'{base_dir}/final_models/optimal_config_{model_name}_{xv_mode}_{experiment_name}'
			f'_pix{pixel}_bs{batch_size}.csv'))

	all_config = [
			optimal_configs.iloc[i] for i in range(optimal_configs.shape[0])
	]
	hash_string = optimal_configs[
		(optimal_configs.random_seed ==random_seed)]['hash'].tolist()[0]


	all_predictions = []
	for pskew in [0.1, 0.5, 0.9]:
			print(pskew)
			pskew_predictions = get_pred(pskew, random_seed, fixed_joint, aux_joint_skew,
				hash_string, base_dir)
			all_predictions.append(pskew_predictions)

	all_predictions = pd.concat(all_predictions, ignore_index = True)
	all_predictions['model'] = f'{model_name}_{xv_mode}'

	all_predictions.to_csv(
		(f'{base_dir}/final_models/optimal_pred_{model_name}_{xv_mode}_{experiment_name}'
			f'_pix{pixel}_bs{batch_size}.csv'),
	index=False)


def get_pred_for_random_seed(random_seed, pixel, batch_size,
	fixed_joint, aux_joint_skew, model_name, xv_mode, experiment_name,
	base_dir, **args):

	# -- get the optimal model configs
	optimal_configs = pd.read_csv(
			(f'{base_dir}/final_models/optimal_config_{model_name}_{xv_mode}_{experiment_name}'
			f'_pix{pixel}_bs{batch_size}.csv'))
	all_config = [
			optimal_configs.iloc[i] for i in range(optimal_configs.shape[0])
	]
	optimal_hash_string = optimal_configs[(optimal_configs.random_seed ==random_seed)]['hash'].tolist()[0]


	# --- get all the configs
	all_config = configurator.get_sweep(experiment_name, model_name, batch_size)

	available_configs = [
		utils.tried_config(config, base_dir=base_dir) for config in all_config
	]
	all_config = list(itertools.compress(all_config, available_configs))
	print(f'Found {len(all_config)} trained models.')


	all_model_predictions = []
	for config in all_config:
		hash_string = utils.config_hasher(config)
		all_predictions = []
		for pskew in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]:
				print(pskew)
				pskew_predictions = get_pred(pskew, random_seed, fixed_joint, aux_joint_skew,
					hash_string, base_dir)
				all_predictions.append(pskew_predictions)

		all_predictions = pd.concat(all_predictions, ignore_index = True)
		all_predictions['hash'] = hash_string
		all_predictions['optimal'] = hash_string == optimal_hash_string
		all_model_predictions.append(all_predictions)

		all_model_predictions_df = pd.concat(all_model_predictions, ignore_index=True)
		all_model_predictions_df['model'] = f'{model_name}_{xv_mode}'
		all_model_predictions_df.to_csv(
			(f'{base_dir}/final_models/all_pred_{model_name}_{xv_mode}_{experiment_name}'
				f'_pix{pixel}_bs{batch_size}.csv'),
		index=False)


if __name__ == "__main__":

	implemented_models = open(
		f'{Path(__file__).resolve().parent}/implemented_models.txt',
		"r").read().split("\n")

	parser = argparse.ArgumentParser()

	parser.add_argument('--base_dir', '-base_dir',
		help="Base directory where the final model will be saved",
		type=str)


	parser.add_argument('--experiment_name', '-experiment_name',
		default='unskew_train',
		choices=['unskew_train', 'skew_train'],
		help="Which experiment to run",
		type=str)


	parser.add_argument('--random_seed', '-random_seed',
		help="Random seed for which we want to get predictions",
		type=int)

	parser.add_argument('--batch_size', '-batch_size',
		help="batch size",
		type=int)

	parser.add_argument('--pixel', '-pixel',
		help="pixels",
		type=int)

	parser.add_argument('--model_name', '-model_name',
		default='unweighted_baseline',
		choices=implemented_models,
		help="Which model to predict for",
		type=str)

	parser.add_argument('--xv_mode', '-xv_mode',
		default='classic',
		choices=['classic', 'two_step'],
		help=("which cross validation algorithm do you want to get preds for"),
		type=str)

	parser.add_argument('--fixed_joint', '-fixed_joint',
		action='store_true',
		default=True,
		help="test set has fixed joint for aux?")

	parser.add_argument('--aux_joint_skew', '-aux_joint_skew',
		help="the joint probability for the aux labels, only matters if fixed_joint is true",
		default = -1.0,
		type=float)


	parser.add_argument('--get_optimal_only', '-get_optimal_only',
		action='store_true',
		default=True,
		help="get all predictions or predictions of the optimal model?")


	args = vars(parser.parse_args())
	if args['get_optimal_only']:
		get_optimal_pred_for_random_seed(**args)
	else:
		assert 1==2
		get_pred_for_random_seed(**args)

