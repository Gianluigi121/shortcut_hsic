"""Chexpert weighting methods."""
import numpy as np
import pandas as pd

def get_simple_weights(data):
	# --- load data
	data = data['0'].str.split(",", expand=True)
	D = data.shape[0]
	data.columns = ['file_name'] +  [f'y{i}' for i in range(D-1)]

	for i in range(D-1): 
		data[f'y{i}'] = data[f'y{i}'].astype(np.float32)

	data['weights'] = 0.0

	# --- compute weights
	for y0_val in [0, 1]:
			for y1_val in [0, 1]:
					for y2_val in [0, 1]:
							mask = (data.y0 == y0_val) * (data.y1 == y1_val) * (data.y2 == y2_val) * 1.0
							denom = np.mean(mask)
							num = np.mean((data.y0 == y0_val)) * np.mean((data.y1 == y1_val) * (data.y2 == y2_val))
							data['weights'] = mask * (num/denom) + (1 - mask) * data['weights']
	txt_data = data.file_name 
	for i in range(D -1):
		txt_data = txt_data + ',' + data[f'y{i}'].astype(str)
	
	txt_data = txt_data + ',' + data.weights.astype(str)

	txt_data = txt_data.apply(lambda x: [x])
	return txt_data

