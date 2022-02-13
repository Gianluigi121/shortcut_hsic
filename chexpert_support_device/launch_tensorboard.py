""" Script for launching tensorboard for multiple networks """
import shared.train_utils as utils
import collections
import subprocess
import shutil, glob

if __name__ == "__main__":
	move_old = False
	scratch_dir = '/scratch/mmakar_root/mmakar0/mmakar/multiple_shortcut/chexpert/tuning'
	bashCommand = 'tensorboard --port 8088 --host localhost --logdir_spec '

	for alpha in [0.0, 100.0, 1000.0, 10000.0]:
		for sigma in [10.0, 100.0, 1000.0]:
			if (alpha == 0.0) and sigma !=10.0:
				continue
			param_dict = {
				'random_seed': 0,
				'pixel': 128,
				'l2_penalty': 0.0,
				'embedding_dim': -1,
				'sigma': sigma,
				'alpha': alpha,
				"architecture": "pretrained_densenet",
				"batch_size": 64,
				'weighted': 'False',
				"conditional_hsic": 'False',
				"skew_train": 'True',
				'num_epochs': 50
			}

			config = collections.OrderedDict(sorted(param_dict.items()))
			hash_string = utils.config_hasher(config)
			bashCommand = bashCommand + (
				f'a{alpha}_s{sigma}:{scratch_dir}/{hash_string},'
			)

	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
