"""Chexpert weighting methods."""
import numpy as np
import pandas as pd

def get_simple_weights(data):
	# --- load data
	data = data['0'].str.split(",", expand=True)
	D = data.shape[1] -1
	data.columns = ['file_name'] +  [f'y{i}' for i in range(D)]

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
			num =  py * pv
			data['weights'] = mask * (num/denom) + (1 - mask) * data['weights']

	txt_data = data.file_name
	for i in range(D):
		txt_data = txt_data + ',' + data[f'y{i}'].astype(str)

	txt_data = txt_data + ',' + data.weights.astype(str)

	txt_data = txt_data.apply(lambda x: [x])
	return txt_data

