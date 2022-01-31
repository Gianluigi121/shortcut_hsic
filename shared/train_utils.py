"""Utility functions to support the main training algorithm"""
import glob
import os
import tensorflow as tf


def restrict_GPU_tf(gpuid, memfrac=0, use_cpu=False):
	""" Function to pick the gpu to run on
		Args:
			gpuid: str, comma separated list "0" or "0,1" or even "0,1,3"
			memfrac: float, fraction of memory. By default grows dynamically
	"""
	if not use_cpu:
		os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

		config = tf.compat.v1.ConfigProto()
		if memfrac == 0:
			config.gpu_options.allow_growth = True
		else:
			config.gpu_options.per_process_gpu_memory_fraction = memfrac
		tf.compat.v1.Session(config=config)
		print("Using GPU:{} with {:.0f}% of the memory".format(gpuid, memfrac * 100))
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = ""
		os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

		print("Using CPU")


def cleanup_directory(directory):
	"""Deletes all files within directory and its subdirectories.

	Args:
		directory: string, the directory to clean up
	Returns:
		None
	"""
	if os.path.exists(directory):
		files = glob.glob(f"{directory}/*", recursive=True)
		for f in files:
			if os.path.isdir(f):
				os.system(f"rm {f}/* ")
			else:
				os.system(f"rm {f}")


def flatten_dict(dd, separator='_', prefix=''):
	""" Flattens the dictionary with eval metrics """
	return {
		prefix + separator + k if prefix else k: v
		for kk, vv in dd.items()
		for k, v in flatten_dict(vv, separator, kk).items()
	} if isinstance(dd,
		dict) else {prefix: dd}