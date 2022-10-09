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


	# this is all the variables 
	# 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
	# 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
	# 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
	# 'Support Devices'],


	# these are distinguishable from pnumonia 
	aux_vars = ['Support Devices']

	drop_vars = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 
	 'Fracture', 'Lung Opacity','Edema', 'Consolidation', 'Pneumothorax',
		'Atelectasis', 'Pleural Effusion', 'Pleural Other', 'Lung Lesion']

	for var in drop_vars: 
		df = df[((df[var] !=1) & (df[var] !=-1))]
		print(var, df.shape)

	# -- clean up a bit
	df['patient'] = df.Path.str.extract(r'(patient)(\d+)')[1]
	df['study'] = df.Path.str.extract(r'(study)(\d+)')[1].astype(int)
	df['uid'] = df['patient'] + "_" + df['study'].astype(str)
	df = df[
		['uid', 'patient', 'study', 'Sex', 'Age', 'Frontal/Lateral',
		'Pneumonia'] + aux_vars  + ['Path']
	]

	# get the main outcome
	df['y0'] = df['Pneumonia'].copy()
	df.y0.fillna(0, inplace=True)
	df.y0[(df.y0 == -1)] = 1

	# print(f'Lost {100*(current_uid - len(df.uid.unique()))/current_uid:.3f}% because of unknown sex')
	# current_uid = len(df.uid.unique())
	df['y1'] = df['Support Devices'].copy()
	df.y1.fillna(0, inplace=True)
	df.y1[(df.y1 == -1)] = 1
	df.drop('Support Devices', axis=1, inplace=True)


	# get the third auxiliary label
	df = df[(df.Sex != 'Unknown')]
	df['y2'] = (df.Sex == 'Male').astype(int)
	df.drop('Sex', axis=1, inplace=True)


	df['y3'] = (df.Age> 30).astype(int)
	df.drop('Age', axis=1, inplace=True)

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
		['Frontal/Lateral', 'frontal', 'Pneumonia'],
		axis=1, inplace=True)
	# print(df.head())

	for i in range(1, 4):
		print(i)
		print(df[['y0', f'y{i}']].groupby(f'y{i}').mean())

	print(df.columns)
	# save
	df.to_csv(f'{save_directory}/penumonia_nofinding_sd_cohort.csv', index=False)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--save_directory', '-save_directory',
		help="Directory where the final cohort will be saved",
		type=str)

	args = vars(parser.parse_args())
	main(**args)
