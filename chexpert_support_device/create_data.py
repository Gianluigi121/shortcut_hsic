"""Creates data for support device+sex shortcut Chexpert."""
import os
import argparse
import pandas as pd
import numpy as np

from copy import deepcopy
pd.set_option('mode.chained_assignment', None)


def main(save_directory):
	""" Function that creates a csv with the cohort"""
	# -- import the data
	if os.path.isdir('/data/ddmg/slabs/CheXpert-v1.0'):
		data_dir = '/data/ddmg/slabs/CheXpert-v1.0'
	elif os.path.isdir('/nfs/turbo/coe-rbg/CheXpert-v1.0'): 
		data_dir = '/nfs/turbo/coe-rbg/CheXpert-v1.0'
	elif os.path.isdir('/nfs/turbo/coe-soto/CheXpert-v1.0'): 
		data_dir = '/nfs/turbo/coe-soto/CheXpert-v1.0'

	else: 
		raise ValueError("cant find data!")

	trdf = pd.read_csv(f'{data_dir}/train.csv')
	vdf = pd.read_csv(f'{data_dir}/valid.csv')
	df = trdf.append(vdf)
	del trdf, vdf

	# Index(['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 'No Finding',
	#        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
	#        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
	#        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
	#        'Support Devices'],


	second_var = 'Enlarged Cardiomediastinum'
	# -- keep only healthy, pneumonia and support device
	df = df[(
		(df['No Finding'] == 1) | (df['Pneumonia'] == 1) | (df[second_var] == 1)
	)]

	# print(df.Pneumonia.value_counts(dropna=False, normalize=True))
	# print(df[second_var].value_counts(dropna=False))

	# -- clean up a bit
	df['patient'] = df.Path.str.extract(r'(patient)(\d+)')[1]
	df['study'] = df.Path.str.extract(r'(study)(\d+)')[1].astype(int)
	df['uid'] = df['patient'] + "_" + df['study'].astype(str)
	df = df[
		['uid', 'patient', 'study', 'Sex', 'Frontal/Lateral',
		'Pneumonia', second_var, 'Path']
	]

	# current_uid = len(df.uid.unique())
	# print(f'Total uids {current_uid}')

	# get the main outcome
	df['y0'] = df['Pneumonia'].copy()
	df.y0.fillna(0, inplace=True)
	df.y0[(df.y0 == -1)] = 1

	# print(df.y0.value_counts(dropna=False, normalize=True))

	# get the first auxiliary label
	df = df[(df.Sex != 'Unknown')]
	df['y1'] = (df.Sex == 'Male').astype(int)
	df.drop('Sex', axis=1, inplace=True)

	# print(f'Lost {100*(current_uid - len(df.uid.unique()))/current_uid:.3f}% because of unknown sex')
	# current_uid = len(df.uid.unique())

	# get the second auxiliary label
	df['y2'] = df[second_var].copy()
	df.y2.fillna(0, inplace=True)
	df.y2[(df.y2 == -1)] = 1
	# print(df.y2.value_counts(dropna = False, normalize = True))

	# keep only studies with frontal views
	df['frontal'] = (df['Frontal/Lateral'] == 'Frontal').astype(int)
	df = df[(df.frontal == 1)]

	# more clean ups
	# print(f'Lost {100*(current_uid - len(df.uid.unique()))/current_uid:.3f}% because they dont have frontal views')
	# current_uid = len(df.uid.unique())

	df.drop_duplicates(subset=['uid'], inplace=True)
	# print(f'Lost {100*(current_uid - df.shape[0])/current_uid:.3f}% because they have duplicates')
	# current_uid = len(df.uid.unique())

	df.drop(
		['Frontal/Lateral', 'frontal', 'Pneumonia', second_var],
		axis=1, inplace=True)
	# print(df.head())

	print(df[['y0','y1', 'y2']].groupby(['y1', 'y2']).agg('mean'))

	# save
	df.to_csv(f'{save_directory}/penumonia_nofinding_sd_cohort.csv', index=False)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--save_directory', '-save_directory',
		help="Directory where the final cohort will be saved",
		type=str)

	args = vars(parser.parse_args())
	main(**args)
