"""Chexpert weighting methods."""

def get_simple_weights(data):
	# --- load data
	data = data['0'].str.split(",", expand=True)
	data.columns = ['file_name', 'y0', 'y1', 'y2']

	data['y0'] = data.y0.astype(np.float32)
	data['y1'] = data.y1.astype(np.float32)
	data['y2'] = data.y2.astype(np.float32)
	data['weights'] = 0.0

	# --- compute weights
	for y0_val in [0, 1]:
			for y1_val in [0, 1]:
					for y2_val in [0, 1]:
							mask = (data.y0 == y0_val) * (data.y1 == y1_val) * (data.y2 == y2_val) * 1.0
							denom = np.mean(mask)
							num = np.mean((data.y0 == y0_val)) * np.mean((data.y1 == y1_val))  * np.mean((data.y2 == y2_val))
							data['weights'] = mask * (num/denom) + (1 - mask) * data['weights']
	data = data.file_name + \
			',' + data.y0.astype(str) + \
			',' + data.y1.astype(str) + \
			',' + data.y2.astype(str) + \
			',' + data.weights.astype(str)

	data = data.apply(lambda x: [x])
	return data

