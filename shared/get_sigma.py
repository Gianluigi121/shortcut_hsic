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

"""Creates config dictionaries for different experiments and models waterbirds"""
import os
import functools
from pathlib import Path
from random import sample
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing
import tqdm
from copy import deepcopy
from shared import weighting as wt

tf.autograph.set_verbosity(0)

import chexpert_support_device.data_builder as chx
import waterbirds.data_builder as wb
import dr.data_builder as dr
import shared.train_utils as utils
from shared import evaluation

def get_last_saved_model(estimator_dir):
	subdirs = [x for x in Path(estimator_dir).iterdir()
		if x.is_dir() and 'temp' not in str(x)]
	try:
		latest_model_dir = str(sorted(subdirs)[-1])
		loaded = tf.saved_model.load(latest_model_dir)
		model = loaded.signatures["serving_default"]
	except:
		print(estimator_dir)
	return model

def get_data_dr_ttest(config, base_dir, n_permute):
	# experiment_directory = (f"{base_dir}/experiment_data/rs{config['random_seed']}"
	# 	f"_v_dim{config['v_dim']}")
	experiment_directory = f"{base_dir}/experiment_data/rs{config['random_seed']}"

	if 'alg_step' not in config.keys():
		_, valid_data, _ = dr.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			alg_step='None')

	else:
		raise NotImplementedError("need to implement this")
		_, valid_data, _ = dr.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			alg_step=config['alg_step'])

	downsample = False
	if downsample:
		validation_data = pd.read_csv(
			f'{experiment_directory}/valid.txt')

		if config['weighted'] == 'True':
			validation_data = wt.get_permutation_weights(validation_data,
				'dr', 'tr_consistent', return_df=True)
			validation_data['group'] = validation_data.y0.map(str) + "-" + validation_data.y1.map(str)
			validation_data = validation_data.groupby('group').sample(
				frac=0.2, random_state=0).reset_index(drop=True)

			txt_data = validation_data.img_name + ',' + validation_data.noise_img
			# TODO dont hard code
			for i in range(2):
				txt_data = txt_data + ',' + validation_data[f'y{i}'].astype(str)

			txt_data = txt_data.apply(lambda x: [x])
			txt_data = txt_data.values.tolist()
			txt_data = [
				tuple(txt_data[i][0].split(',')) for i in range(len(txt_data))
			]
			valid_data = txt_data
		else:
			raise NotImplementedError("not yet")


	map_to_image_label_given_pixel = functools.partial(dr.map_to_image_label,
		pixel=config['pixel'], weighted=config['weighted'])

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	batch_size = int(len(valid_data) / n_permute)
	valid_dataset = valid_dataset.batch(batch_size,
		drop_remainder=True).repeat(1)
	return valid_dataset


def get_data_dr_perm(config, base_dir):
	# experiment_directory = (f"{base_dir}/experiment_data/rs{config['random_seed']}"
	# 	f"_v_dim{config['v_dim']}")
	experiment_directory = f"{base_dir}/experiment_data/rs{config['random_seed']}"

	if 'alg_step' not in config.keys():
		_, valid_data, _ = dr.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			alg_step='None')

	else:
		raise NotImplementedError("need to implement this")
		_, valid_data, _ = dr.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			alg_step=config['alg_step'])

	downsample = False
	if downsample:
		validation_data = pd.read_csv(
			f'{experiment_directory}/valid.txt')

		if config['weighted'] == 'True':
			validation_data = wt.get_permutation_weights(validation_data,
				'dr', 'tr_consistent', return_df=True)
			validation_data['group'] = validation_data.y0.map(str) + "-" + validation_data.y1.map(str)
			validation_data = validation_data.groupby('group').sample(
				frac=0.2, random_state=0).reset_index(drop=True)

			txt_data = validation_data.img_name + ',' + validation_data.noise_img
			# TODO dont hard code
			for i in range(2):
				txt_data = txt_data + ',' + validation_data[f'y{i}'].astype(str)

			txt_data = txt_data.apply(lambda x: [x])
			txt_data = txt_data.values.tolist()
			txt_data = [
				tuple(txt_data[i][0].split(',')) for i in range(len(txt_data))
			]
			valid_data = txt_data
		else:
			raise NotImplementedError("not yet")


	map_to_image_label_given_pixel = functools.partial(dr.map_to_image_label,
		pixel=config['pixel'], weighted=config['weighted'])

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	valid_dataset = valid_dataset.batch(config['batch_size'],
		drop_remainder=True).repeat(1)
	return valid_dataset


def get_data_waterbirds_perm(config, base_dir):
	# experiment_directory = (f"{base_dir}/experiment_data/rs{config['random_seed']}"
	# 	f"_v_dim{config['v_dim']}")
	experiment_directory = f"{base_dir}/experiment_data/rs{config['random_seed']}"

	if 'alg_step' not in config.keys():
		_, valid_data, _ = wb.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			v_dim=config['v_dim'], alg_step='None')

	else:
		raise NotImplementedError("need to implement this")
		_, valid_data, _ = wb.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			v_dim=config['v_dim'], alg_step=config['alg_step'])

	map_to_image_label_given_pixel = functools.partial(wb.map_to_image_label,
		pixel=config['pixel'], weighted=config['weighted'])

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	valid_dataset = valid_dataset.batch(config['batch_size'],
		drop_remainder=False).repeat(1)
	return valid_dataset


def get_data_waterbirds_ttest(config, base_dir, n_permute):
	# experiment_directory = (f"{base_dir}/experiment_data/rs{config['random_seed']}"
	# 	f"_v_dim{config['v_dim']}")
	experiment_directory = f"{base_dir}/experiment_data/rs{config['random_seed']}"

	if 'alg_step' not in config.keys():
		_, valid_data, _ = wb.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			v_dim=config['v_dim'], alg_step='None')

	else:
		raise NotImplementedError("need to implement this")
		_, valid_data, _ = wb.load_created_data(
			experiment_directory=experiment_directory, weighted=config['weighted'],
			v_dim=config['v_dim'], alg_step=config['alg_step'])

	map_to_image_label_given_pixel = functools.partial(wb.map_to_image_label,
		pixel=config['pixel'], weighted=config['weighted'])

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	batch_size = int(len(valid_data) / n_permute)
	valid_dataset = valid_dataset.batch(batch_size,
		drop_remainder=True).repeat(1)
	return valid_dataset




def get_data_chexpert_perm(config, base_dir):
	experiment_directory = f"{base_dir}/experiment_data/rs{config['random_seed']}"

	if 'alg_step' not in config.keys():
		_, valid_data, _ = chx.load_created_data(
			experiment_directory=experiment_directory, 
			skew_train=config['skew_train'],
			weighted=config['weighted'],
			v_mode=config['v_mode'], 
			v_dim=config['v_dim'], 
			alg_step='None')
	else:
		raise NotImplementedError("not yet")
		_, valid_data, _ = chx.load_created_data(
			chexpert_data_dir=base_dir, random_seed=config['random_seed'],
			v_mode=config['v_mode'], v_dim=config['v_dim'],
			skew_train=config['skew_train'], weighted=config['weighted'],
			alg_step=config['alg_step'])

	map_to_image_label_given_pixel = functools.partial(chx.map_to_image_label,
		pixel=config['pixel'], weighted=config['weighted'])

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	valid_dataset = valid_dataset.batch(config['batch_size'],
		drop_remainder=False).repeat(1)
	return valid_dataset

def get_optimal_sigma_for_run_ttest(config, base_dir, t1_error, n_permute=3):
	# TODO: need to pass this as an argument
	# -- get the dataset
	if 'chexpert' in base_dir:
		raise NotImplementedError("not yet")
		valid_dataset = get_data_chexpert_ttest(config, base_dir)
	elif 'waterbirds' in base_dir:
		valid_dataset = get_data_waterbirds_ttest(config, base_dir, n_permute)
	else:
		valid_dataset = get_data_dr_ttest(config, base_dir, n_permute)

	# -- model
	hash_string = utils.config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')
	model = get_last_saved_model(hash_dir)

	metric_values = []
	# ---compute hsic over folds
	for batch_id, examples in enumerate(valid_dataset):
		# print(f'{batch_id} / {n_permute}')
		x, labels_weights = examples
		sample_weights = labels_weights['sample_weights']
		labels = labels_weights['labels']
		zpred = model(tf.convert_to_tensor(x))['embedding']
		hsic_val = evaluation.hsic(
			x=zpred, y=labels[:, 1:],
			sample_weights=sample_weights,
			sigma=config['sigma'])[[0]].numpy()
		metric_values.append(hsic_val)

	metric_values = np.hstack(metric_values)
	curr_results = pd.DataFrame({
		'random_seed': config['random_seed'],
		'alpha': config['alpha'],
		'sigma': config['sigma'],
		'hsic': hsic_val,
		'significant': stats.ttest_1samp(metric_values, 0.0)[1] # lower values --> reject
		# 'significant': stats.wilcoxon(metric_values)[1]
	}, index=[0])
	if (np.mean(metric_values) == 0.0 and np.var(metric_values) == 0.0):
		curr_results['significant'] = 1

	return curr_results


def get_optimal_sigma_for_run_perm(config, base_dir, t1_error, n_permute=100):

	# TODO: need to pass this as an argument
	# -- get the dataset
	if 'chexpert' in base_dir:
		valid_dataset = get_data_chexpert_perm(config, base_dir)
	elif 'waterbirds' in base_dir:
		valid_dataset = get_data_waterbirds_perm(config, base_dir)
	else:
		valid_dataset = get_data_dr_perm(config, base_dir)

	# -- model
	hash_string = utils.config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')
	model = get_last_saved_model(hash_dir)

	# ---compute hsic over folds
	z_pred_list = []
	labels_list = []
	sample_weights_list = []
	for batch_id, examples in enumerate(valid_dataset):
		# print(f'{batch_id} / {n_permute}')
		x, labels_weights = examples
		sample_weights = labels_weights['sample_weights']
		sample_weights_list.append(sample_weights)

		labels = labels_weights['labels']
		labels_list.append(labels)

		zpred = model(tf.convert_to_tensor(x))['embedding']
		z_pred_list.append(zpred)

	zpred = tf.concat(z_pred_list, axis=0)
	labels = tf.concat(labels_list, axis=0)
	sample_weights = tf.concat(sample_weights_list, axis=0)

	hsic_val = evaluation.hsic(
		x=zpred, y=labels[:, 1:],
		sample_weights=sample_weights,
		sigma=config['sigma'])[[0]].numpy()

	perm_hsic_val = []
	for seed in range(n_permute):
		# if seed % 10 ==0:
		# 	print(f'{seed}/{n_permute}')
		labels_p = labels.numpy()
		np.random.RandomState(seed).shuffle(labels_p)
		labels_p = tf.constant(labels_p)
		# labels_p = tf.random.shuffle(labels, seed=seed)
		# labels_p = tf.cast(tf.random.stateless_binomial(
		# 	shape=tf.shape(labels), probs=0.5, counts=1,
		# 	seed=[seed, seed]), dtype=tf.float32)

		perm_hsic_val.append(evaluation.hsic(
			x=zpred, y=labels_p[:, 1:],
			sample_weights=sample_weights,
			sigma=config['sigma'])[[0]].numpy())

	perm_hsic_val = np.concatenate(
		perm_hsic_val, axis=0)

	thresh = np.quantile(perm_hsic_val, 1 - t1_error)
	accept_null = hsic_val <= thresh
	print(config['random_seed'], config['sigma'], config['alpha'], hsic_val, thresh, accept_null[0])

	curr_results = pd.DataFrame({
		'random_seed': config['random_seed'],
		'alpha': config['alpha'],
		'sigma': config['sigma'],
		'hsic': hsic_val,
		'significant': accept_null
	}, index=[0])

	perm_vals = pd.DataFrame(perm_hsic_val).transpose()
	perm_vals.columns = [f'hsicp{i}' for i in range(len(perm_hsic_val))]
	curr_results = pd.concat([curr_results, perm_vals], axis=1)
	return curr_results

def get_optimal_sigma(all_config, t1_error, n_permute,
	num_workers, base_dir):
	mode = 'perm'
	if 'dr' in base_dir:
		filename = f"all_hsic_{mode}_epoch{all_config[0]['num_epochs']}.csv"
	else:
		filename = f"all_hsic_{mode}_vdim{all_config[0]['v_dim']}.csv"

	if 'dr' in base_dir:
		if os.path.exists(f'{base_dir}/final_models/{filename}'):
			all_config_df = pd.DataFrame(all_config)
			all_config_df = all_config_df[['random_seed', 'alpha', 'sigma']]
			all_results = pd.read_csv(f'{base_dir}/final_models/{filename}')
			print(f"Found {all_results.shape[0]} cached permutations")

			print(all_config_df.shape[0])
			all_results = all_results.merge(all_config_df,
				on=['random_seed', 'alpha', 'sigma'], how='right',
				indicator=True)
			if 'right_only' in all_results._merge.unique().tolist():
				remaining_config_ids = all_results[
					(all_results._merge == 'right_only')].index.tolist()
				if len(remaining_config_ids) > 0:
					all_config = [all_config[i] for i in remaining_config_ids]
				else:
					all_config = []
			else:
				all_results['quant'] = np.quantile(
					all_results[[col for col in all_results.columns if "hsicp" in col]].values,
					1-t1_error, axis=1)
				all_results['significant'] = all_results.hsic <= all_results.quant
				return all_results.drop(
					[f'hsicp{i}' for i in range(n_permute)] + ['quant'], axis=1)

			all_results = all_results[(all_results._merge == "both")]
			all_results.drop('_merge', axis=1, inplace=True)
			old_results = all_results.copy()

	if len(all_config) > 0:
		all_results = []

		if mode == 'perm':
			runner_wrapper = functools.partial(get_optimal_sigma_for_run_perm,
				base_dir=base_dir, t1_error=t1_error, n_permute=n_permute)
		else:
			runner_wrapper = functools.partial(get_optimal_sigma_for_run_ttest,
				base_dir=base_dir, t1_error=t1_error, n_permute=n_permute)

		if num_workers <= 0:
			for cid, config in enumerate(all_config):
				print(cid)
				results = runner_wrapper(config)
				all_results.append(results)
		else:
			pool = multiprocessing.Pool(num_workers)
			for results in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config),
				total=len(all_config)):
				all_results.append(results)

		all_results = pd.concat(all_results, axis=0, ignore_index=True)

		try:
			all_results = pd.concat([all_results, old_results], axis=0,
				ignore_index=True)
		except:
			print("no existing results added")

		if 'dr' in base_dir:
			all_results.to_csv(f'{base_dir}/final_models/{filename}',
				index=False)

	if mode == 'perm':
		all_results['quant'] = np.quantile(
			all_results[[col for col in all_results.columns if "hsicp" in col]].values,
			1 - t1_error, axis=1)
		all_results['significant'] = all_results.hsic <= all_results.quant

		all_results.drop(
			[f'hsicp{i}' for i in range(n_permute)] + ['quant'], axis=1, inplace=True)
	return all_results





