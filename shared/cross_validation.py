""" Script for cross validaiton """
import os
import pickle
import pandas as pd
import functools
import multiprocessing
import tqdm

from shared.train_utils import config_hasher, tried_config_file
from shared import get_sigma

def import_helper(config, base_dir):
	"""Imports the dictionary with the results of an experiment.

	Args:
		args: tuple with model, config where
			model: str, name of the model we're importing the performance of
			config: dictionary, expected to have the following: exp_dir, the experiment
				directory random_seed,  random seed for the experiment py1_y0_s,
				probability of y1=1| y0=1 in the shifted test distribution alpha,
				hsic/cross prediction penalty sigma,  kernel bandwidth for the hsic penalty
				l2_penalty,  regularization parameter dropout_rate,  drop out rate
				embedding_dim,  dimension of the final representation/embedding
				unused_kwargs, other key word args passed to xmanager but not needed here

	Returns:
		pandas dataframe of results if the file was found, none otherwise
	"""
	if config is None:
		return
	hash_string = config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string)
	performance_file = os.path.join(hash_dir, 'performance.pkl')

	if not os.path.exists(performance_file):
		logging.error('Couldnt find %s', performance_file)
		return None

	results_dict = pickle.load(open(performance_file, 'rb'))
	results_dict.update(config)

	results_dict['hash'] = hash_string
	return pd.DataFrame(results_dict, index=[0])


def import_results(configs, num_workers, base_dir):

	import_helper_wrapper = functools.partial(import_helper, base_dir=base_dir)
	pool = multiprocessing.Pool(num_workers)
	res = []
	for config_res in tqdm.tqdm(pool.imap_unordered(import_helper_wrapper,
		configs), total=len(configs)):
		res.append(config_res)
	res = pd.concat(res, axis=0, ignore_index=True, sort=False)
	return res, configs


def reshape_results(results):
	shift_columns = [col for col in results.columns if 'shift' in col]
	shift_metrics_columns = [
		col for col in shift_columns if ('pred_loss' in col) or ('accuracy' in col) or ('auc' in col)
	]
	# shift_metrics_columns = shift_metrics_columns + [
	# 	f'shift_{py}_hsic' for py in [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]]

	results = results[shift_metrics_columns]
	results = results.transpose()

	results['py1_y0_s'] = results.index.str[6:10]
	results['py1_y0_s'] = results.py1_y0_s.str.replace('_', '')
	results['py1_y0_s'] = results.py1_y0_s.astype(float)

	results_auc = results[(results.index.str.contains('auc'))]
	results_auc = results_auc.rename(columns={
		col: f'auc_{col}' for col in results_auc.columns if col != 'py1_y0_s'
	})
	results_loss = results[(results.index.str.contains('pred_loss'))]
	results_loss = results_loss.rename(columns={
		col: f'loss_{col}' for col in results_loss.columns if col != 'py1_y0_s'
	})

	results_final = results_auc.merge(results_loss, on=['py1_y0_s'])

	print(results_final)
	return results_final



def get_optimal_model_results(mode, configs, base_dir, hparams,
	 num_workers, t1_error, n_permute):

	if mode not in ['classic', 'two_step']:
		raise NotImplementedError('Can only run classic or two_step modes')
	if mode == 'classic':
		return get_optimal_model_classic(configs, None, base_dir, hparams, num_workers)
	elif mode =='two_step':
		return get_optimal_model_two_step(configs, base_dir, hparams,
			t1_error, n_permute, num_workers)



def get_optimal_model_two_step(configs, base_dir, hparams, t1_error,
	n_permute, num_workers):
	all_results, available_configs = import_results(configs, num_workers, base_dir)
	sigma_results = get_sigma.get_optimal_sigma(available_configs, t1_error=t1_error,
		n_permute=n_permute, num_workers=num_workers, base_dir=base_dir)
	print("this is sig res")
	print(sigma_results.sort_values(['random_seed', 'sigma', 'alpha']))

	if not sigma_results.significant.max():
		sigma_results.significant[(sigma_results.hsic == sigma_results.hsic.min())] = True

	sigma_results = sigma_results[(sigma_results.significant==True)]
	filtered_results = all_results.merge(sigma_results, on=['random_seed', 'sigma', 'alpha'])

	filtered_results.drop(['significant'], inplace=True, axis=1)
	filtered_results.reset_index(drop=True, inplace=True)

	unique_filtered_results = filtered_results[['random_seed', 'sigma', 'alpha']].copy()
	unique_filtered_results.drop_duplicates(inplace=True)

	return get_optimal_model_classic(None, filtered_results, base_dir, hparams, num_workers)



def get_optimal_model_classic(configs, filtered_results, base_dir, hparams, num_workers):
	if ((configs is None) and (filtered_results is None)):
		raise ValueError("Need either configs or table of results_dict")

	if configs is not None:
		print("getting results")
		all_results, _ = import_results(configs, num_workers, base_dir)
	else:
		all_results = filtered_results.copy()

	columns_to_keep = hparams + ['random_seed', 'validation_pred_loss']
	best_loss = all_results[columns_to_keep]
	print(print(best_loss.sort_values(['random_seed', 'sigma', 'alpha'])))
	best_loss = best_loss.groupby('random_seed').validation_pred_loss.min()
	best_loss = best_loss.to_frame()
	best_loss.reset_index(drop=False, inplace=True)
	best_loss.rename(columns={'validation_pred_loss': 'min_validation_pred_loss'},
		inplace=True)
	all_results = all_results.merge(best_loss, on='random_seed')

	all_results = all_results[
		(all_results.validation_pred_loss == all_results.min_validation_pred_loss)
	]

	print(all_results[['random_seed', 'sigma', 'alpha', 'l2_penalty']])

	optimal_configs = all_results[['random_seed', 'hash']]

	# --- get the final results over all runs
	mean_results = all_results.mean(axis=0).to_frame()
	mean_results.rename(columns={0: 'mean'}, inplace=True)
	std_results = all_results.std(axis=0).to_frame()
	std_results.rename(columns={0: 'std'}, inplace=True)
	final_results = mean_results.merge(
		std_results, left_index=True, right_index=True
	)

	final_results = final_results.transpose()
	final_results_clean = reshape_results(final_results)
	return final_results_clean, optimal_configs



