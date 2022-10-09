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

"""Runs the full correlation sweep for the waterbirds experiment."""
import functools
import itertools
import subprocess
import multiprocessing
import os
import pickle
import argparse
import tqdm
from pathlib import Path

import shared.train_utils as utils
from waterbirds import configurator

# TODO: need to manage overwriting better
UM_USER = 'mmakar'

if UM_USER == 'precisionhealth':
	UM_SCRATCH_DIR = '/scratch/precisionhealth_owned_root/precisionhealth_owned1'
	ACCOUNT = 'precisionhealth_owned1'
	PARTITION = 'precisionhealth'

if UM_USER == 'mmakar':
	UM_SCRATCH_DIR = '/scratch/mmakar_root/mmakar0/'
	ACCOUNT = 'mmakar0'
	PARTITION = 'gpu'


ARMIS_MAIN_DIR = '/nfs/turbo/coe-rbg'
GL_MAIN_DIR = '/nfs/turbo/coe-soto'
MIT_MAIN_DIR = '/data/ddmg/scate/'

if os.path.isdir(GL_MAIN_DIR):
	MAIN_DIR = GL_MAIN_DIR
	SCRATCH_DIR = UM_SCRATCH_DIR
	HOST = 'GL'
	PARTITION = 'gpu'

elif os.path.isdir(ARMIS_MAIN_DIR):
	MAIN_DIR = ARMIS_MAIN_DIR
	SCRATCH_DIR = UM_SCRATCH_DIR
	HOST = 'ARMIS'

elif os.path.isdir(MIT_MAIN_DIR):
	MAIN_DIR = MIT_MAIN_DIR
	SCRATCH_DIR = MIT_SCRATCH_DIR
	HOST = 'TIG'
else:
	raise ValueError(("Can't locate data resources, please point to"
	" the right directory that has the data"))


def runner(config, base_dir, checkpoint_dir, slurm_save_dir, overwrite,
	submit):
	"""Trains model in config if not trained before.
	Args:
		config: dict with config
	Returns:
		Nothing
	"""
	# check system status
	hash_string = utils.config_hasher(config)
	print(hash_string)
	model_dir = os.path.join(base_dir, 'tuning', hash_string)
	if not os.path.exists(model_dir):
		os.system(f'mkdir -p {model_dir}')

	checkpoint_dir = os.path.join(checkpoint_dir, 'tuning', hash_string)
	if not os.path.exists(checkpoint_dir):
		os.system(f'mkdir -p {checkpoint_dir}')

	pickle.dump(config, open(f'{model_dir}/config.pkl', 'wb'))
	config['data_dir'] = base_dir
	config['exp_dir'] = model_dir
	config['checkpoint_dir'] = checkpoint_dir
	config['gpuid'] = '$CUDA_VISIBLE_DEVICES'

	flags = ' '.join('--%s %s \\\n' % (k, str(v)) for k, v in config.items())
	if os.path.exists(f'{slurm_save_dir}/{hash_string}.sbatch'):
		os.remove(f'{slurm_save_dir}/{hash_string}.sbatch')
	f = open(f'{slurm_save_dir}/{hash_string}.sbatch', 'x')
	f.write('#!/bin/bash\n')
	f.write('#SBATCH --time=10:00:00\n')
	f.write('#SBATCH --cpus-per-task=2\n')
	f.write('#SBATCH --output=gpu.out\n')
	f.write('#SBATCH --gres=gpu:1\n')
	# f.write('#SBATCH --gpus-per-task=1\n')
	if HOST == 'ARMIS':
		f.write(f'#SBATCH --account={ACCOUNT}\n')
		f.write(f'#SBATCH --partition={PARTITION}\n')
		f.write(f'#SBATCH --mail-user=mmakar@umich.edu\n')
		f.write(f'#SBATCH --mail-type=BEGIN,END\n')
		# f.write('#SBATCH -w, --nodelist=armis28004\n')
		# f.write('#SBATCH --mem-per-gpu=20000m\n')
	if HOST == 'GL':
		f.write(f'#SBATCH --account={ACCOUNT}\n')
		f.write(f'#SBATCH --partition={PARTITION}\n')
		f.write(f'#SBATCH --mail-user=mmakar@umich.edu\n')
		f.write(f'#SBATCH --mail-type=BEGIN,END\n')
		f.write('#SBATCH --mem-per-gpu=90000m\n')
	if HOST == 'TIG':
		f.write('#SBATCH -w, --nodelist=tig-slurm-2\n')
		f.write('#SBATCH --partition=gpu\n')
		f.write('#SBATCH --mem=40000m\n')
	# first check if there is any room on nfs
	f.write(f'''nfs_amount_used=$(df {MAIN_DIR}'''
		''' | awk '{printf sub(/%.*/, "")}')\n'''
		'''if [ "$nfs_amount_used" -gt 95 ]\n'''
		'''then\n'''
		'''	echo "Not enough memory on NFS"\n'''
		'''	exit\n'''
		'''fi\n''')
	# second check if there is any room on the scratch folder
	f.write(f'''scratch_amount_used=$(df {SCRATCH_DIR} '''
		'''| awk '{printf sub(/%.*/, "")}')\n'''
		'''if [ "$scratch_amount_used" -gt 95 ]\n'''
		'''then\n'''
		'''	echo "Not enough memory on Scratch"\n'''
		'''	exit\n'''
		'''fi\n''')
	if not overwrite:
		f.write(f'if [ ! -f "{model_dir}/performance.pkl" ]; then\n')
	f.write(f'	python -m waterbirds.main {flags} > {model_dir}/log.log 2>&1 \n')
	if not overwrite:
		f.write('fi\n')
	f.close()

	if submit:
		subprocess.call(f'sbatch --dependency=afterany:314685 {slurm_save_dir}/{hash_string}.sbatch',
			shell=True)


def main(base_dir,
					checkpoint_dir,
					slurm_save_dir,
					model_to_tune,
					v_dim,
					batch_size,
					overwrite,
					submit,
					num_workers,
					clean_directories):
	"""Main function to tune/train the model.
	Args:
		experiment_name: str, name of the experiemnt to run
		base_dir: str, the directory where the final model will be saved
		checkpoint_dir: str, the directory where the checkpoints will be saved
		slurm_save_dir: str, the directory where the generated slurm scripts will
			be saved
		model_to_tune: str, which model to tune/train
		num_trials: int, number of hyperparams to train for
		overwrite: bool, whether or not to retrain if a specific hyperparam config
			has already been tried
		submit: bool, whether or not to submit on slurm
		num_workers: int. number of cpus to use to run the script generation in parallel
		clean_directors: boolean. if true erases all existing trained models
		Returns:
			nothing
	"""
	if not os.path.exists(slurm_save_dir):
		os.system(f'mkdir -p {slurm_save_dir}')
	all_config = configurator.get_sweep(model_to_tune,
		v_dim, batch_size)
	print(f'All configs are {len(all_config)}')
	if not overwrite:
		configs_to_consider = [
			not utils.tried_config(config, base_dir=base_dir) for config in all_config
		]
		all_config = list(itertools.compress(all_config, configs_to_consider))

	print(f'Remaining configs are {len(all_config)}')

	# all_config = all_config[:1]
	runner_wrapper = functools.partial(runner, base_dir=base_dir,
		checkpoint_dir=checkpoint_dir, slurm_save_dir=slurm_save_dir,
		overwrite=overwrite, submit=submit)

	if num_workers > 0:
		pool = multiprocessing.Pool(num_workers)
		for _ in tqdm.tqdm(pool.imap_unordered(runner_wrapper, all_config),
			total=len(all_config)):
			pass

	else:
		for config in all_config:
			runner_wrapper(config)

	if clean_directories:
		raise NotImplementedError("havent implemented cleaning up yet")


if __name__ == "__main__":

	implemented_models = open(
		f'{Path(__file__).resolve().parent}/implemented_models.txt',
		"r").read().split("\n")

	parser = argparse.ArgumentParser()


	parser.add_argument('--base_dir', '-base_dir',
		help="Base directory where the final model will be saved",
		type=str)

	parser.add_argument('--checkpoint_dir', '-checkpoint_dir',
		help="Checkpoint directory where the model checkpoints will be saved",
		type=str)

	parser.add_argument('--slurm_save_dir', '-slurm_save_dir',
		help="Directory where the slurm scripts will be saved",
		type=str)

	parser.add_argument('--model_to_tune', '-model_to_tune',
		default='unweighted_baseline',
		choices=implemented_models,
		help="Which model to tune",
		type=str)

	parser.add_argument('--overwrite', '-overwrite',
		action='store_true',
		default=False,
		help="If this config has been tested before, rerun?")

	parser.add_argument('--submit', '-submit',
		action='store_true',
		default=False,
		help="Should I submit the file to run on slurm?")

	parser.add_argument('--num_workers', '-num_workers',
		default=0,
		help=("Number of workers to use in parallel to generate the"
			" slurm scripts. If 0, scripts are not generated in parallel"),
		type=int)

	parser.add_argument('--batch_size', '-batch_size',
		help="batch size",
		type=int)

	parser.add_argument('--v_dim', '-v_dim',
		help="dimension of additional Vs",
		type=int)

	parser.add_argument('--clean_directories', '-clean_directories',
		action='store_true',
		default=False,
		help="NUCLEAR: delete all model results?")

	args = vars(parser.parse_args())
	main(**args)
