""" Cross validation for chexpert support device"""
import os
import argparse
import functools
import itertools
from pathlib import Path
from waterbirds import configurator
import shared.train_utils as utils
import shared.cross_validation as cv


def main(base_dir, model_to_tune,
	xv_method, v_dim, batch_size, num_workers,
	t1_error, n_permute):

	if not os.path.exists(f'{base_dir}/final_models/'):
		os.mkdir(f'{base_dir}/final_models/')

	all_config = configurator.get_sweep(
		model_to_tune, v_dim, batch_size)
	print(f'All configs are {len(all_config)}')
	original_configs = len(all_config)

	# -- Get all the configs that are available
	configs_available = [
		utils.tried_config(config, base_dir=base_dir) for config in all_config
	]

	all_config = list(itertools.compress(all_config, configs_available))
	found_configs = len(all_config)
	print(f'------ FOUND {found_configs} / {original_configs}---------')
	best_model_results, best_model_configs = cv.get_optimal_model_results(
		mode=xv_method,
		configs=all_config,
		base_dir=base_dir,
		hparams=['alpha', 'sigma', 'l2_penalty', 'embedding_dim'],
		num_workers=num_workers,
		t1_error=t1_error,
		n_permute=n_permute)

	if 'v_dim' in all_config[0].keys():
		best_model_results.to_csv(
			(f"{base_dir}/final_models/optimal_results_{model_to_tune}_{xv_method}"
				f"_pix{all_config[0]['pixel']}_"
				f"bs{all_config[0]['batch_size']}_vdim{all_config[0]['v_dim']}.csv"),
			index=False)

		best_model_configs.to_csv(
			(f"{base_dir}/final_models/optimal_config_{model_to_tune}_{xv_method}"
				f"_pix{all_config[0]['pixel']}_"
				f"bs{all_config[0]['batch_size']}_vdim{all_config[0]['v_dim']}.csv"),
			index=False)
	else: 

		best_model_results.to_csv(
			(f"{base_dir}/final_models/optimal_results_{model_to_tune}_{xv_method}"
				f"_pix{all_config[0]['pixel']}_"
				f"bs{all_config[0]['batch_size']}.csv"),
			index=False)

		best_model_configs.to_csv(
			(f"{base_dir}/final_models/optimal_config_{model_to_tune}_{xv_method}"
				f"_pix{all_config[0]['pixel']}_"
				f"bs{all_config[0]['batch_size']}.csv"),
			index=False)

if __name__ == "__main__":
	implemented_models = open(
		f'{Path(__file__).resolve().parent}/implemented_models.txt',
		"r").read().split("\n")

	parser = argparse.ArgumentParser()


	parser.add_argument('--base_dir', '-base_dir',
		help="Base directory",
		type=str)

	parser.add_argument('--model_to_tune', '-model_to_tune',
		default='unweighted_baseline',
		choices=implemented_models,
		help="Which model to xvalidate",
		type=str)

	parser.add_argument('--xv_method', '-xv_method',
		default='classic',
		choices=['classic', 'two_step'],
		help="Which cross validation method?",
		type=str)

	parser.add_argument('--batch_size', '-batch_size',
		help="batch size",
		type=int)

	parser.add_argument('--num_workers', '-num_workers',
		help="number of workers used in parallel",
		type=int)

	parser.add_argument('--t1_error', '-t1_error',
		help="level of tolerance for rejecting null",
		type=float)

	parser.add_argument('--n_permute', '-n_permute',
		help="number of permutations for hsic null test",
		default=100,
		type=int)

	parser.add_argument('--v_dim', '-v_dim',
		help="dimension of additional Vs",
		type=int)
	args = vars(parser.parse_args())
	main(**args)


