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

tf.autograph.set_verbosity(0)

import chexpert_support_device.data_builder as chx
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


def get_data_chexpert(kfolds, config, base_dir):
	experiment_directory = f"{base_dir}/experiment_data/rs{config['random_seed']}"

	_, valid_data, _ = chx.load_created_data(
		chexpert_data_dir=base_dir, random_seed=config['random_seed'],
		skew_train=config['skew_train'], weighted=config['weighted'])

	map_to_image_label_given_pixel = functools.partial(chx.map_to_image_label,
		pixel=config['pixel'], weighted=config['weighted'])

	valid_dataset = tf.data.Dataset.from_tensor_slices(valid_data)
	valid_dataset = valid_dataset.map(map_to_image_label_given_pixel, num_parallel_calls=1)
	batch_size = int(len(valid_data) / kfolds)
	valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True).repeat(1)
	return valid_dataset


def get_optimal_sigma_for_run(config, kfolds, base_dir):
	# -- get the dataset
	valid_dataset = get_data_chexpert(kfolds, config, base_dir)

	# -- model
	hash_string = utils.config_hasher(config)
	hash_dir = os.path.join(base_dir, 'tuning', hash_string, 'saved_model')
	model = get_last_saved_model(hash_dir)

	# ---compute hsic over folds
	metric_values = []
	for batch_id, examples in enumerate(valid_dataset):
		# print(f'{batch_id} / {kfolds}')
		x, labels_weights = examples
		sample_weights = labels_weights['sample_weights']
		labels = labels_weights['labels']

		logits = model(tf.convert_to_tensor(x))['logits']
		zpred = model(tf.convert_to_tensor(x))['embedding']

		metric_value = evaluation.hsic(
			x=zpred, y=labels[:, 1:],
			sample_weights=sample_weights,
			sigma=config['sigma'])[[0]].numpy()

		metric_values.append(metric_value)

	curr_results = pd.DataFrame({
		'random_seed': config['random_seed'],
		'alpha': config['alpha'],
		'sigma': config['sigma'],
		'hsic': np.mean(metric_values),
		'pval': stats.ttest_1samp(metric_values, 0.0)[1]
	}, index=[0])
	if (np.mean(metric_values) == 0.0 and np.var(metric_values) == 0.0):
		curr_results['pval'] = 1
	return curr_results


def get_optimal_sigma(all_config, kfolds, num_workers, base_dir):
	all_results = []
	runner_wrapper = functools.partial(get_optimal_sigma_for_run,
		kfolds=kfolds, base_dir=base_dir)

	if num_workers <=0:
		for cid, config in enumerate(all_config):
			print(cid)
			results = runner_wrapper(config)
			all_results.append(results)
	else:
		pool = multiprocessing.Pool(num_workers)
		for results in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config), total=len(all_config)):
			all_results.append(results)

	all_results = pd.concat(all_results, axis=0, ignore_index=True)
	return all_results





