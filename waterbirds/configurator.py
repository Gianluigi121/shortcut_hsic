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

"""Creates config dictionaries for different models"""

import collections
import itertools
from pathlib import Path
import warnings

def configure_hsic_model(v_dim, weighted, batch_size):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [i for i in range(10)],
		'pixel': [128],
		'l2_penalty': [0.0],
		'embedding_dim': [-1],
		# 'sigma': [100.0, 1000.0],
		# 'alpha': [1e3, 1e5, 1e7, 1e9],
		'sigma': [1000.0],
		'alpha': [1e7],
		"architecture": ["pretrained_resnet"],
		"batch_size": [batch_size],
		'weighted': [weighted],
		"conditional_hsic": ['False'],
		'num_epochs': [500],
		'v_dim': [v_dim]
	}
	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep


def configure_baseline(v_dim, weighted, batch_size):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	if (v_dim !=0 and weighted == 'False'):
		warnings.warn(("Unweighted baseline doesn't utilize aux labels. Setting "
			"v_dim to zero"))
		v_dim = 0

	all_sweep = []
	for rs in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

		# param_dict = {
		# 	'random_seed': [rs],
		# 	'pixel': [128],
		# 	'l2_penalty': [0.0, 0.0001, 0.001],
		# 	# 'l2_penalty': [0.0],
		# 	'embedding_dim': [-1],
		# 	'sigma': [10.0],
		# 	'alpha': [0.0],
		# 	"architecture": ["pretrained_resnet"],
		# 	"batch_size": [batch_size],
		# 	'weighted': ['False'],
		# 	"conditional_hsic": ['False'],
		# 	'num_epochs': [500],
		# 	'v_dim': [0]
		# }

		# print(param_dict)
		# param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
		# keys, values = zip(*param_dict_ordered.items())
		# sweep1 = [dict(zip(keys, v)) for v in itertools.product(*values)]


		param_dict = {
			'random_seed': [rs],
			'pixel': [128],
			'l2_penalty': [0.0, 0.0001, 0.001],
			# 'l2_penalty': [0.0],
			'embedding_dim': [-1],
			'sigma': [10.0],
			'alpha': [0.0],
			"architecture": ["pretrained_resnet"],
			"batch_size": [batch_size],
			'weighted': ['True'],
			"conditional_hsic": ['False'],
			'num_epochs': [500],
			'v_dim': [2]
		}

		param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
		keys, values = zip(*param_dict_ordered.items())
		sweep2 = [dict(zip(keys, v)) for v in itertools.product(*values)]

		param_dict = {
			'random_seed': [rs],
			'pixel': [128],
			'l2_penalty': [0.0, 0.0001, 0.001],
			# 'l2_penalty': [0.0],
			'embedding_dim': [-1],
			'sigma': [10.0],
			'alpha': [0.0],
			"architecture": ["pretrained_resnet"],
			"batch_size": [batch_size],
			'weighted': ['True'],
			"conditional_hsic": ['False'],
			'num_epochs': [500],
			'v_dim': [12]
		}

		param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
		keys, values = zip(*param_dict_ordered.items())
		sweep3 = [dict(zip(keys, v)) for v in itertools.product(*values)]

		param_dict = {
			'random_seed': [rs],
			'pixel': [128],
			'l2_penalty': [0.0],
			'embedding_dim': [-1],
			'sigma': [10.0, 100.0, 1000.0],
			'alpha': [1e3, 1e5, 1e7, 1e9],
			"architecture": ["pretrained_resnet"],
			"batch_size": [batch_size],
			'weighted': ['True'],
			"conditional_hsic": ['False'],
			'num_epochs': [500],
			'v_dim': [2]
		}
		param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
		keys, values = zip(*param_dict_ordered.items())
		sweep4 = [dict(zip(keys, v)) for v in itertools.product(*values)]


		param_dict = {
			'random_seed': [rs],
			'pixel': [128],
			'l2_penalty': [0.0],
			'embedding_dim': [-1],
			'sigma': [10.0, 100.0, 1000.0],
			'alpha': [1e3, 1e5, 1e7, 1e9],
			"architecture": ["pretrained_resnet"],
			"batch_size": [batch_size],
			'weighted': ['True'],
			"conditional_hsic": ['False'],
			'num_epochs': [500],
			'v_dim': [12]
		}
		param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
		keys, values = zip(*param_dict_ordered.items())
		sweep5 = [dict(zip(keys, v)) for v in itertools.product(*values)]

		all_sweep = all_sweep + sweep2 + sweep3 + sweep4 + sweep5
	return all_sweep

def configure_baseline_all(v_dim, weighted, batch_size):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	if (v_dim !=0 and weighted == 'False'):
		warnings.warn(("Unweighted baseline doesn't utilize aux labels. Setting "
			"v_dim to zero"))
		v_dim = 0

	param_dict = {
		'random_seed': [3, 4, 5, 6, 7, 8, 9],
		'pixel': [128],
		'l2_penalty': [0.0, 0.0001, 0.001],
		'embedding_dim': [-1],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["pretrained_resnet"],
		"batch_size": [batch_size],
		'weighted': [weighted],
		"conditional_hsic": ['False'],
		'num_epochs': [500],
		'v_dim': [v_dim]
	}

	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep


def configure_first_step_model(batch_size):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [5],
		'pixel': [128],
		'l2_penalty': [0.0, 0.0001, 0.001],
		'embedding_dim': [-1],
		"architecture": ["pretrained_resnet"],
		"batch_size": [batch_size],
		'num_epochs': [500],
		"alg_step": ['first']
	}
	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep

def get_sweep(model, v_dim, batch_size):
	"""Wrapper function, creates configurations based on experiment and model.

	Args:
		model: string, which model to create the configs for
		aug_prop: float, proportion of augmentation relative to training data.
			Only relevant for augmentation based baselines

	Returns:
		Iterator with all hyperparameter combinations
	"""

	implemented_models = open(
		f'{Path(__file__).resolve().parent}/implemented_models.txt',
		"r").read().split("\n")

	if model not in implemented_models:
		raise NotImplementedError((f'Model {model} parameter configuration'
															' not implemented'))

	if model == 'unweighted_baseline':
		return configure_baseline(v_dim=v_dim,
		 weighted='False', batch_size=batch_size)

	if model == 'weighted_baseline':
		return configure_baseline(v_dim=v_dim, weighted='True',
			batch_size=batch_size)

	if model == 'weighted_bal_baseline':
		return configure_baseline(v_dim=v_dim, weighted='True_bal',
			batch_size=batch_size)


	if model == 'unweighted_hsic':
		return configure_hsic_model(v_dim=v_dim, weighted='False',
			batch_size=batch_size)

	if model == 'weighted_hsic':
		return configure_hsic_model(v_dim=v_dim, weighted='True',
			batch_size=batch_size)
	if model == 'weighted_bal_hsic':
		return configure_hsic_model(v_dim=v_dim, weighted='True_bal',
			batch_size=batch_size)


	if model == 'first_step':
		return configure_first_step_model(
						batch_size=batch_size)




