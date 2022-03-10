import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np 
import os

print(os.getcwd())

trdf = pd.read_csv('/nfs/turbo/coe-rbg/CheXpert-v1.0/train.csv')
vdf = pd.read_csv('/nfs/turbo/coe-rbg/CheXpert-v1.0/valid.csv')
df = trdf.append(vdf)
del trdf, vdf

df = df[((df['No Finding'] == 1) | (df['Pneumonia'] == 1))]

df['patient'] = df.Path.str.extract(r'(patient)(\d+)')[1]
df['study'] = df.Path.str.extract(r'(study)(\d+)')[1].astype(int)
df['uid'] = df['patient'] + "_" + df['study'].astype(str)
df = df[['uid', 'patient', 'study', 'Sex', 'Frontal/Lateral', 'Pneumonia', 'Path', 'Age']]

current_uid = len(df.uid.unique())
print(f'Total uids {current_uid}')

# get the main outcome(Pneumonia)
df['y0'] = df['Pneumonia'].copy()
df.y0.fillna(0, inplace = True)
df.y0[(df.y0 == -1)] = 1
df.y0.value_counts(dropna = False, normalize = True)

# get the auxiliary label(Sex)
df = df[(df.Sex != 'Unknown')]
df['y1'] = (df.Sex == 'Male').astype(int)
df.drop('Sex', axis = 1, inplace = True)

print(f'Lost {100*(current_uid - len(df.uid.unique()))/current_uid:.3f}% because of unknown sex')
current_uid = len(df.uid.unique())

# keep only studies with frontal views
df['frontal'] = (df['Frontal/Lateral'] == 'Frontal').astype(int)
df = df[(df.frontal ==1)]

print(f'Lost {100*(current_uid - len(df.uid.unique()))/current_uid:.3f}% because they dont have frontal views')
current_uid = len(df.uid.unique())

df.drop_duplicates(subset=['uid'], inplace = True)
print(f'Lost {100*(current_uid - df.shape[0])/current_uid:.3f}% because they have duplicates')
current_uid = len(df.uid.unique())

df.drop(['Frontal/Lateral', 'frontal', 'Pneumonia'], axis = 1, inplace = True)
df.head()

age_list = df.Age.unique()
print(f"min age: {min(age_list)}, max age: {max(age_list)}")
print(f"Number of different ages: {len(age_list)}")

print(len(df.uid.unique()))

df.to_csv('/nfs/turbo/coe-rbg/zhengji/age/penumonia_nofinding_cohort.csv', index = False)
