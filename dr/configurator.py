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

def configure_hsic_model(py1y0, weighted, batch_size):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [0],
		'pixel': [299],
		'l2_penalty': [0.0],
		'embedding_dim': [-1],
		'sigma': [100.0, 1000.0],
		'alpha': [1e7, 1e9],
		"architecture": ["pretrained_inception"],
		"batch_size": [batch_size],
		'weighted': [weighted],
		"conditional_hsic": ['False'],
		'num_epochs': [200],
		'py1y0': [py1y0]
	}
	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep


def configure_baseline(py1y0, weighted, batch_size):
	"""Creates hyperparameters for correlations experiment for SLABS model.

	Returns:
		Iterator with all hyperparameter combinations
	"""
	param_dict = {
		'random_seed': [0],
		'pixel': [299],
		'l2_penalty': [0.0, 0.0001, 0.001],
		# 'l2_penalty': [0.0],
		'embedding_dim': [-1],
		'sigma': [10.0],
		'alpha': [0.0],
		"architecture": ["pretrained_inception"],
		"batch_size": [batch_size],
		'weighted': [weighted],
		"conditional_hsic": ['False'],
		'num_epochs': [200],
		'py1y0': [py1y0]
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
		'pixel': [299],
		'l2_penalty': [0.0, 0.0001, 0.001],
		'embedding_dim': [-1],
		"architecture": ["pretrained_inception"],
		"batch_size": [batch_size],
		'num_epochs': [200],
		"alg_step": ['first']
	}
	print(param_dict)
	param_dict_ordered = collections.OrderedDict(sorted(param_dict.items()))
	keys, values = zip(*param_dict_ordered.items())
	sweep = [dict(zip(keys, v)) for v in itertools.product(*values)]

	return sweep

def get_sweep(model, py1y0, batch_size):
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
		return configure_baseline(py1y0=py1y0,
			weighted='False', batch_size=batch_size)

	if model == 'weighted_baseline':
		return configure_baseline(py1y0=py1y0, weighted='True',
			batch_size=batch_size)

	if model == 'weighted_bal_baseline':
		return configure_baseline(py1y0=py1y0, weighted='True_bal',
			batch_size=batch_size)

	if model == 'unweighted_hsic':
		return configure_hsic_model(py1y0=py1y0, weighted='False',
			batch_size=batch_size)

	if model == 'weighted_hsic':
		return configure_hsic_model(py1y0=py1y0, weighted='True',
			batch_size=batch_size)
	if model == 'weighted_bal_hsic':
		return configure_hsic_model(py1y0=py1y0, weighted='True_bal',
			batch_size=batch_size)

	if model == 'first_step':
		return configure_first_step_model(
			batch_size=batch_size)




