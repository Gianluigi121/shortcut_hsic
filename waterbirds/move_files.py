from waterbirds import configurator
import shared.train_utils as utils
import itertools
import os, shutil
import pickle


if __name__ == "__main__":
	base_dir = '/data/ddmg/scate/multiple_shortcut/waterbirds'
	model_to_tune = 'unweighted_baseline'
	v_dim = 0 
	batch_size = 64
	filepath = '/data/ddmg/scate/multiple_shortcut/waterbirds/tuning'
	
	# all_files = [f for f in os.listdir(filepath)]
	# n_all_files = len(all_files)

	# # ---- all configs 
	# all_config = configurator.get_sweep(model_to_tune,
	# 	 v_dim, batch_size)

	# configs_tried = [
	# 	utils.tried_config(config, base_dir=base_dir) for config in all_config
	# ]
	# all_config = list(itertools.compress(all_config, configs_tried))


	# config_files = []
	# for config in all_config:
	# 	hash_string = utils.config_hasher(config)
	# 	config_files.append(hash_string)

	# n_config_files = len(config_files)

	# files_to_remove = list(set(all_files) - set(config_files))

	# print(n_all_files, n_config_files, len(files_to_remove))

	# for config_file_rm in files_to_remove:
	# 	shutil.rmtree(f'{filepath}/{config_file_rm}')

	# ---- all configs 
	all_config = configurator.get_sweep(model_to_tune,
		 v_dim, batch_size)

	configs_tried = [
		utils.tried_config(config, base_dir=base_dir) for config in all_config
	]
	all_config = list(itertools.compress(all_config, configs_tried))

	for config_id, config in enumerate(all_config):
		print(config['v_dim'])
		# orig_hash = utils.config_hasher(config)
		# new_v_dim = 2 if config['v_dim'] == 0 else 12

		# # update the config pickle file
		# config_pkl = pickle.load(open(f'{filepath}/{orig_hash}/config.pkl', 'rb'))
		# config_pkl['v_dim'] = new_v_dim
		# pickle.dump(config_pkl, open(f'{filepath}/{orig_hash}/config.pkl', 'wb'))

		# # rename directory
		# config['v_dim'] = new_v_dim
		# new_hash = utils.config_hasher(config)

		# shutil.move(f'{filepath}/{orig_hash}', f'{filepath}/{new_hash}')

