"""weighting methods."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.chained_assignment = None

def get_binary_weights(data, data_type, weighting_type):
	# --- load data
	data = data['0'].str.split(",", expand=True)

	if data_type == 'chexpert':
		D = data.shape[1] -2
		data.columns = ['file_name', 'noise_img'] + [f'y{i}' for i in range(D)]
	elif data_type == 'waterbirds':
		D = data.shape[1] - 4
		data.columns = ['bird_img', 'bird_seg', 'back_img', 'noise_img'] + \
			[f'y{i}' for i in range(D)]

	for i in range(D):
		data[f'y{i}'] = data[f'y{i}'].astype(np.float32)


	data['weights'] = 0.0

	# --- get all combinations
	all_y_vals = data[[f'y{i}' for i in range(D)]]
	all_y_vals.drop_duplicates(inplace=True)
	all_y_vals = all_y_vals.values
	# --- compute weights
	for i in range(all_y_vals.shape[0]):
		mask = data[[f'y{i}' for i in range(D)]] == all_y_vals[i,:]
		mask = mask.min(axis=1)
		denom = np.mean(mask)
		if denom == 0:
			data['weights'] = mask * 0.0 + (1 - mask) * data['weights']
		else:
			py = np.mean((data['y0'] == all_y_vals[i,0]))
			pv = data[[f'y{i}' for i in range(1, D)]] == all_y_vals[i, 1:]
			pv = pv.min(axis=1)
			pv = np.mean(pv)
			if weighting_type == "tr_consistent":
				num =  py * pv
			else:
				num = pv
			data['weights'] = mask * (num/denom) + (1 - mask) * data['weights']

	# print(data.weights.isnull().sum())
	# print(data.weights.min(), data.weights.max(), data.weights.mean(), data.weights.var())
	if data_type == 'chexpert':
		txt_data = data.file_name

	elif data_type == 'waterbirds':
		txt_data = data.bird_img + \
			',' + data.bird_seg + \
			',' + data.back_img + \
			',' + data.noise_img

	for i in range(D):
		txt_data = txt_data + ',' + data[f'y{i}'].astype(str)

	txt_data = txt_data + ',' + data.weights.astype(str)

	txt_data = txt_data.apply(lambda x: [x])
	return txt_data


def get_permutation_weights(data, data_type, weighting_type):
	rng = np.random.RandomState(0)
	# --- load data
	data = data['0'].str.split(",", expand=True)
	if data_type == 'chexpert':
		D = data.shape[1] -2
		data.columns = ['img_name', 'noise_img'] + [f'y{i}' for i in range(D)]
	elif data_type == 'dr':
		D = data.shape[1] - 2
		data.columns = ['img_name', 'noise_img'] + [f'y{i}' for i in range(D)]

	elif data_type == 'waterbirds':
		D = data.shape[1] - 4
		data.columns = ['bird_img', 'bird_seg', 'back_img', 'noise_img'] + \
			[f'y{i}' for i in range(D)]

	for i in range(D):
		data[f'y{i}'] = data[f'y{i}'].astype(np.float32)

	# --- create permuted dataset
	obs_data = data[[f'y{i}' for i in range(D)]].copy()
	obs_data['label'] = 0

	perm_data = data[[f'y{i}' for i in range(D)]].copy()
	perm_y0 = perm_data['y0'].values
	perm_y0 = rng.choice(perm_y0, size=len(perm_y0), replace=False)
	perm_data['y0'] = perm_y0
	perm_data['label'] = 1

	weighting_data = pd.concat([obs_data, perm_data], axis=0)

	# clf = LogisticRegression(penalty='none')
	clf = RandomForestClassifier(random_state=0)
	clf.fit(weighting_data[[f'y{i}' for i in range(D)]],
		weighting_data.label)
	data['weights'] = clf.predict_proba(data[[f'y{i}' for i in range(D)]])[:, 1]
	data['weights'] = data['weights'] / (1.0 - data['weights'])

	if data_type == 'chexpert':
		txt_data = data.img_name + \
			',' + data.noise_img		
	if data_type == 'dr':
		txt_data = data.img_name + \
			',' + data.noise_img
	elif data_type == 'waterbirds':
		txt_data = data.bird_img + \
			',' + data.bird_seg + \
			',' + data.back_img + \
			',' + data.noise_img

	for i in range(D):
		txt_data = txt_data + ',' + data[f'y{i}'].astype(str)

	txt_data = txt_data + ',' + data.weights.astype(str)

	txt_data = txt_data.apply(lambda x: [x])
	return txt_data

