""" Script to get the predictions for the cross validated model."""

import os
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import functools
import multiprocessing
import tqdm
import itertools

import tensorflow as tf
import dr.data_builder as db
from dr import configurator
import shared.train_utils as utils

def get_test_data(config, dist, base_dir):
	"""Function to get the data."""
	experiment_directory = (
		f"{base_dir}/experiment_data/rs{config['random_seed']}")

	test_data = pd.read_csv(
		f'{experiment_directory}/test_{dist}.txt'
	).values.tolist()

	test_data = [
		tuple(test_data[i][0].split(',')) for i in range(len(test_data))
	]

	map_to_image_label_given_pixel = functools.partial(
		db.map_to_image_label_test, pixel=config['pixel'],
		weighted=config['weighted'])

	test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
	test_dataset = test_dataset.map(map_to_image_label_given_pixel,
		num_parallel_calls=1)
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


def get_pred(hash_string, dist, base_dir):
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')
	model = get_last_saved_model(hash_dir)

	config_dir = os.path.join(base_dir, 'tuning', hash_string, 'config.pkl')
	config = pickle.load(open(config_dir, 'rb'))

	test_dataset = get_test_data(
		config=config,
		dist=dist,
		base_dir=base_dir
	)

	pred_df_list = []
	for batch_id, examples in enumerate(test_dataset):
		if batch_id % 50 ==0:
			print(f'{batch_id}')
		x, labels_weights = examples
		predictions = model(tf.convert_to_tensor(x))['probabilities']

		pred_df = pd.DataFrame(labels_weights['labels'].numpy())
		pred_df.columns = ['y0'] + [f'y{i}' for i in range(1,
			pred_df.shape[1])]

		pred_mat = predictions.numpy()
		pred_df[
			[f'predictions{i}' for i in range(pred_mat.shape[1])]] = pred_mat
		pred_df['pred_class'] = np.argmax(pred_mat, axis=1)

		pred_df_list.append(pred_df)

	pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True)
	return pred_df

def get_optimal_pred_for_random_seed(random_seed, pixel, batch_size,
	 model_name, xv_mode, py1y0, base_dir, **args):

	# -- get the optimal model configs
	optimal_configs = pd.read_csv(
			(f'{base_dir}/final_models/optimal_config_{model_name}_{xv_mode}'
			f'_pix{pixel}_bs{batch_size}_py1y0{py1y0}.csv'))
	all_config = [
			optimal_configs.iloc[i] for i in range(optimal_configs.shape[0])
	]
	optimal_hash_string = optimal_configs[(optimal_configs.random_seed ==random_seed)]['hash'].tolist()[0]

	all_predictions = []
	# for dist in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
	for dist in [0.1, 0.5, 0.9]:
		pred_df = get_pred(optimal_hash_string, dist, base_dir)
		pred_df['model'] = f'{model_name}_{xv_mode}'
		pred_df['dist'] = dist
		all_predictions.append(pred_df)

	all_predictions = pd.concat(all_predictions, ignore_index=True)
	all_predictions.to_csv(
		(f'{base_dir}/final_models/opt_pred_rs{random_seed}_{model_name}_{xv_mode}'
			f'_pix{pixel}_bs{batch_size}_py1y0{py1y0}.csv'),
	index=False)

if __name__ == "__main__":

	implemented_models = open(
		f'{Path(__file__).resolve().parent}/implemented_models.txt',
		"r").read().split("\n")

	parser = argparse.ArgumentParser()

	parser.add_argument('--base_dir', '-base_dir',
		help="Base directory where the final model will be saved",
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

	parser.add_argument('--py1y0', '-py1y0',
		help="training P(V=1 | Y=0)",
		default=0.9,
		type=float)

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

	parser.add_argument('--get_optimal_only', '-get_optimal_only',
		action='store_true',
		default=True,
		help="get all predictions or predictions of the optimal model?")


	args = vars(parser.parse_args())
	if args['get_optimal_only']:
		get_optimal_pred_for_random_seed(**args)
	else:
		raise NotImplementedError("not yet!")

