"""weighting methods."""
import numpy as np
import pandas as pd

def get_binary_weights(data, data_type):
	# --- load data
	data = data['0'].str.split(",", expand=True)
	
	if data_type == 'chexpert':
		D = data.shape[1] -1
		data.columns = ['file_name'] +  [f'y{i}' for i in range(D)]
	elif data_type == 'waterbirds':
		D = data.shape[1]-4
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
			num =  py * pv
			data['weights'] = mask * (num/denom) + (1 - mask) * data['weights']

	print(data.weights.min(), data.weights.max(), data.weights.mean(), data.weights.var())
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

