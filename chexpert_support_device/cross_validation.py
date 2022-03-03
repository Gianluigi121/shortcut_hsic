""" Cross validation for chexpert support device"""
import os
import argparse
import functools
import itertools
from pathlib import Path
from chexpert_support_device import configurator
import shared.train_utils as utils
import shared.cross_validation as cv


def main(base_dir, experiment_name, model_to_tune,
	xv_method, batch_size, num_workers, pval):

	if not os.path.exists(f'{base_dir}/final_models/'):
		os.mkdir(f'{base_dir}/final_models/')

	all_config = configurator.get_sweep(experiment_name, model_to_tune, batch_size)
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
		pval=0.05)

	best_model_results.to_csv(
		(f"{base_dir}/final_models/optimal_results_{model_to_tune}_{xv_method}_{experiment_name}"
			f"_pix{all_config[0]['pixel']}_bs{all_config[0]['batch_size']}.csv"),
		index=False)

	best_model_configs.to_csv(
		(f"{base_dir}/final_models/optimal_config_{model_to_tune}_{xv_method}_{experiment_name}"
			f"_pix{all_config[0]['pixel']}_bs{all_config[0]['batch_size']}.csv"),
		index=False)



if __name__ == "__main__":
	implemented_models = open(
		f'{Path(__file__).resolve().parent}/implemented_models.txt',
		"r").read().split("\n")

	parser = argparse.ArgumentParser()


	parser.add_argument('--base_dir', '-base_dir',
		help="Base directory",
		type=str)

	parser.add_argument('--experiment_name', '-experiment_name',
		default='skew_train',
		choices=['unskew_train', 'skew_train'],
		help="Which experiment to run",
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


	parser.add_argument('--pval', '-pval',
		help="P-value for hsic significance",
		type=float)

	args = vars(parser.parse_args())
	main(**args)


