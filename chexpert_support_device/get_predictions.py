""" Script to get the predictions for the cross validated model."""

import os
import argparse
import pandas as pd
import pickle
import functools
import multiprocessing
import tqdm
import itertools


import tensorflow as tf
import chexpert_support_device.data_builder as db
from chexpert_support_device import configurator
import shared.train_utils as utils
from shared import weighting as wt
from pathlib import Path


def get_test_data(alg_step, config, pskew, base_dir):
	if alg_step == 'None':
		return get_test_data_full(config, pskew, base_dir)
	else:
		return get_test_data_subset(config, base_dir)


def get_test_data_subset(config, base_dir):
	"""Function to get the data."""
	experiment_directory = (
		f"{base_dir}/experiment_data/rs{config['random_seed']}")

	test_data = pd.read_csv(
		f'{experiment_directory}/train.txt'
	)

	idx_dict = pickle.load(
		open(f'{experiment_directory}/first_step.pkl',
	'rb'))

	test_data = test_data.iloc[idx_dict['test_idx']]
	test_data = test_data.values.tolist()

	test_data = [
		tuple(test_data[i][0].split(',')) for i in range(len(test_data))
	]

	map_to_image_label_given_pixel = functools.partial(db.map_to_image_label_test,
		pixel=config['pixel'], weighted=config['weighted'])
	test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
	test_dataset = test_dataset.map(map_to_image_label_given_pixel, 
		num_parallel_calls=1)
	test_dataset = test_dataset.batch(config['batch_size'],
		drop_remainder=False).repeat(1)
	return test_dataset


def get_test_data_full(config, pskew, base_dir):
		"""Function to get the data."""
		experiment_directory = (
			f"{base_dir}/experiment_data/rs{config['random_seed']}")

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


def get_valid_data(config, base_dir):
	"""Function to get the data."""
	experiment_directory = (
		f"{base_dir}/experiment_data/rs{config['random_seed']}")

	if config['v_mode'] == 'normal':
		v_str =''
	else: 
		assert 1==2 
	validation_data = pd.read_csv(
		f"{experiment_directory}/{v_str}{config['skew_str']}_valid.txt")

	if config['weighted'] == 'True':
		validation_data = wt.get_permutation_weights(validation_data,
			'chexpert_sd', 'tr_consistent')
	elif config['weighted'] == 'True_bal':
		validation_data = wt.get_permutation_weights(validation_data,
			'chexpert_sd', 'tr_consistent')

	validation_data = validation_data.values.tolist()
	validation_data = [
		tuple(validation_data[i][0].split(',')) for i in range(len(validation_data))
	]

	# its ok to use the test version here, you're only getting predictions 
	map_to_image_label_given_pixel = functools.partial(db.map_to_image_label_test,
			pixel=config['pixel'], weighted=config['weighted'])
	valid_dataset = tf.data.Dataset.from_tensor_slices(validation_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, 
		num_parallel_calls=1)
	valid_dataset = valid_dataset.batch(config['batch_size'],
		drop_remainder=False).repeat(1)
	return valid_dataset


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


def get_pred(pskew, hash_string, eval_group, base_dir):
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')
	model = get_last_saved_model(hash_dir)

	config_dir = os.path.join(base_dir, 'tuning', hash_string, 'config.pkl')
	config = pickle.load(open(config_dir, 'rb'))

	if 'alg_step' in config.keys():
		if config['alg_step'] == 'first':
			print("need to update this")
			assert 1==2
			alg_step = 'first'
			pred_names = [f'pred{i}' for i in range(1, config['v_dim'] + 1)]
	else:
		alg_step = 'None'
		pred_names = ['pred0']

	if eval_group == 'test':
		test_dataset = get_test_data(
			config=config,
			pskew=pskew,
			base_dir=base_dir
		)
	else: 
		test_dataset = get_valid_data(
			config=config, 
			base_dir=base_dir)

	pred_df_list = []
	for batch_id, examples in enumerate(test_dataset):
		print(f'{batch_id}')
		x, labels_weights = examples
		labels_df = pd.DataFrame(labels_weights['labels'].numpy())
		labels_df.columns = [f'y{i}' for i in range(labels_df.shape[1])]

		predictions = model(tf.convert_to_tensor(x))['probabilities']
		pred_df = pd.DataFrame(predictions.numpy())
		pred_df.columns = pred_names

		pred_df = pd.concat([pred_df, labels_df], axis=1)
		if eval_group == 'valid' and config['weighted'] == 'True':
			pred_df['sample_weights'] = labels_weights['sample_weights'].numpy()

		pred_df_list.append(pred_df)

	pred_df = pd.concat(pred_df_list, axis=0, ignore_index=True)
	return pred_df

def get_all_pred_helper(config, base_dir,
	existing_pred, eval_group):
	hash_string = utils.config_hasher(config)
	if existing_pred is not None:
		pred_exists = hash_string in existing_pred.model.unique().tolist()
	else:
		pred_exists = False

	dist_list = [0.1, 0.5, 0.9] if eval_group == 'test' else [-1]
	if not pred_exists:
		hash_res = []
		for pskew in dist_list:
			curr_pred = get_pred(pskew, hash_string, eval_group, base_dir)
			curr_pred['dist'] = pskew
			hash_res.append(curr_pred)
		hash_res = pd.concat(hash_res, ignore_index=True)
		hash_res['model'] = hash_string
	else:
		hash_res = existing_pred[(existing_pred.model == hash_string)].reset_index(
			drop=True)
	return hash_res


def get_all_pred(experiment_name, model_name, xv_mode, 
	v_mode, v_dim, batch_size, pixel, eval_group, 
	base_dir, n_jobs, **args):
	all_config = configurator.get_sweep(experiment_name, model_name,
		v_mode, v_dim, batch_size)

	filename = (f'{base_dir}/final_models/all_{eval_group}_pred_'
		f'{model_name}_{xv_mode}_pix{pixel}_bs{batch_size}_'
		f'vdim{v_dim}.csv')

	# try:
	# 	existing_pred = pd.read_csv(filename)
	# except:
	# 	existing_pred = None

	existing_pred = None
	available_configs = [
		utils.tried_config(config, base_dir=base_dir) for config in all_config
	]
	all_config = list(itertools.compress(all_config, available_configs))
	print(f'Found {len(all_config)} trained models.')

	all_predictions = []

	if n_jobs > 1:
		pool = multiprocessing.Pool(n_jobs)

		get_pred_for_random_seed_helper_wrapper = functools.partial(
			get_all_pred_helper,
			base_dir=base_dir, existing_pred=existing_pred,
			eval_group=eval_group)
		for curr_pred in tqdm.tqdm(pool.imap_unordered(
			get_pred_for_random_seed_helper_wrapper,
			all_config), total=len(all_config)):
			all_predictions.append(curr_pred)

	else:
		for config in all_config:
			curr_pred = get_all_pred_helper(
				config=config,
				base_dir=base_dir,
				existing_pred=existing_pred,
				eval_group=eval_group)
			all_predictions.append(curr_pred)

	all_predictions = pd.concat(all_predictions, ignore_index=True)
	all_predictions.to_csv(filename, index=False)


def get_optimal_pred_for_random_seed(random_seed, pixel, batch_size,
	 model_name, xv_mode, v_dim, v_mode, experiment_name,
	base_dir, **args):
	# -- get the optimal model configs
	optimal_configs = pd.read_csv(
			(f'{base_dir}/final_models/optimal_config_{model_name}_{xv_mode}_{experiment_name}'
			f'_pix{pixel}_bs{batch_size}_v_dim{v_dim}_v_mode{v_mode}.csv'))

	optimal_hash_string = optimal_configs[
		(optimal_configs.random_seed == random_seed)]['hash'].tolist()[0]

	all_predictions = []
	for pskew in [0.1, 0.5, 0.9]:
			print(f' ====  pskew {pskew} ==== ')
			pskew_predictions = get_pred(pskew, optimal_hash_string, 'test', base_dir)
			all_predictions.append(pskew_predictions)

	all_predictions = pd.concat(all_predictions, ignore_index = True)
	all_predictions['model'] = f'{model_name}_{xv_mode}'

	all_predictions.to_csv(
		(f'{base_dir}/final_models/optimal_pred_{model_name}_{xv_mode}_{experiment_name}'
			f'_pix{pixel}_bs{batch_size}_v_dim{v_dim}_v_mode{v_mode}.csv'),
	index=False)


def get_optimal_pred(seed_list, pixel, batch_size,
		model_name, xv_mode, v_dim, v_mode, experiment_name,
		base_dir, n_jobs, **args):
	seed_list = [int(i) for i in seed_list.split(",")]

	if n_jobs < 1: 
		for random_seed in seed_list:
			print(f'======= Random seed :{random_seed} ========')
			get_optimal_pred_for_random_seed(
				random_seed=random_seed,
				pixel=pixel,
				batch_size=batch_size,
				model_name=model_name,
				xv_mode=xv_mode,
				v_dim=v_dim,
				v_mode=v_mode, 
				experiment_name=experiment_name, 
				base_dir=base_dir)
	else:
		get_optimal_pred_for_rs_wrapper = functools.partial(
			get_optimal_pred_for_random_seed,
			pixel=pixel,
			batch_size=batch_size,
			model_name=model_name,
			xv_mode=xv_mode,
			v_dim=v_dim,
			v_mode=v_mode, 
			experiment_name=experiment_name, 
			base_dir=base_dir)

	pool = multiprocessing.Pool(n_jobs)
	for _ in tqdm.tqdm(pool.imap_unordered(get_optimal_pred_for_rs_wrapper,
	seed_list), total=len(seed_list)):
		pass

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

