import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import pandas as pd
import os, shutil
import functools
import datetime
from copy import deepcopy

MAIN_DIR = "/nfs/turbo/coe-rbg/zhengji/single_shortcut/chexpert/"

"""
Data preparation
"""

path = MAIN_DIR + 'age_data'
if not os.path.exists(path):
    os.mkdir(path)

"""
Assume y_value = 0
Dominant probability: P(y0=y_value, y1=y_value) = len(dominant group) / len(group)
Small probability: P(y0=0, y1=1-y_value) = len(small group) / len(group)

len(dominant group) 
= len(dominant group) / len(small group) * len(small group)
= dominant probability / small probability * len(small_group)

len(small group)
= len(small group) / len(dominant group) * len(dominant group)
= small probability / dominant probability * len(dominant group)
"""
def sample_y2_on_y1(df, y0_value, y1_value, dominant_probability, rng):
	dominant_group = df.index[((df.y0==y0_value) & (df.y1==y1_value) & (df.y2==y1_value))]
	small_group = df.index[((df.y0==y0_value) & (df.y1==y1_value) & (df.y2==(1-y1_value)))]
	
	small_probability = 1 - dominant_probability 

	# CASE I: Smaller group too large, Dominant group too small
	# If the dominant group is smaller than dominant probability*len(group)
	# Truncate the size of the small group based on the dominant probability
	if len(dominant_group) < (dominant_probability/small_probability)*len(small_group):
		dominant_id = deepcopy(dominant_group).tolist()
		small_id = rng.choice(
			small_group,size = int(
				(small_probability/dominant_probability)* len(dominant_group)
			),
			replace = False).tolist()

	# CASE II: Dominant group too large, smaller group too small
	# If the small group if smaller than small probability*len(group)
	# Truncate the size of the large group based on the small probability
	elif len(small_group) < (small_probability/dominant_probability)*len(dominant_group):
		small_id = deepcopy(small_group).tolist()
		dominant_id = rng.choice(
			dominant_group, size = int(
				(dominant_probability/small_probability)*len(small_group)
			), replace = False).tolist()
	new_ids = small_id + dominant_id
	df_new = df.iloc[new_ids]
	return df_new

def sample_y1_on_main(df, y_value, dominant_probability, rng):
	dominant_group = df.index[((df.y0==y_value) & (df.y1 ==y_value))]
	small_group = df.index[((df.y0==y_value) & (df.y1 ==(1-y_value)))]
	
	small_probability = 1 - dominant_probability 
	# CASE I: Smaller group too large, Dominant group too small
	if len(dominant_group) < (dominant_probability/small_probability)*len(small_group):
		dominant_id = deepcopy(dominant_group).tolist()
		small_id = rng.choice(
			small_group,size = int(
				(small_probability/dominant_probability)* len(dominant_group)
			),
			replace = False).tolist()

	# CASE II: Dominant group too large, smaller group too small
	elif len(small_group) < (small_probability/dominant_probability)*len(dominant_group):
		small_id = deepcopy(small_group).tolist()
		dominant_id = rng.choice(
			dominant_group, size = int(
				(dominant_probability/small_probability)*len(small_group)
			), replace = False).tolist()
	new_ids = small_id + dominant_id
	df_new = df.iloc[new_ids]
	return df_new

def fix_marginal(df, y0_probability, rng):
	y0_group = df.index[(df.y0 == 0)]
	y1_group = df.index[(df.y0 == 1)]
	
	y1_probability = 1 - y0_probability 
	if len(y0_group) < (y0_probability/y1_probability) * len(y1_group):
		y0_ids = deepcopy(y0_group).tolist()
		y1_ids = rng.choice(
			y1_group, size = int((y1_probability/y0_probability) * len(y0_group)),
			replace = False).tolist()
	elif len(y1_group) < (y1_probability/y0_probability) * len(y0_group):
		y1_ids = deepcopy(y1_group).tolist()
		y0_ids = rng.choice(
			y0_group, size = int( (y0_probability/y1_probability)*len(y1_group)), 
			replace = False
		).tolist()
	dff = df.iloc[y1_ids + y0_ids]
	dff.reset_index(inplace = True, drop=True)
	reshuffled_ids = rng.choice(dff.index, size = len(dff.index), replace=False).tolist()
	dff = dff.iloc[reshuffled_ids].reset_index(drop = True)
	return dff

def get_skewed_data(cand_df, py1d=0.9, py2d=0.9, py00=0.7, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    # --- Fix the conditional distributions of y2
    cand_df11 = sample_y2_on_y1(cand_df, 1, 1, py2d, rng)
    cand_df10 = sample_y2_on_y1(cand_df, 1, 0, py2d, rng)
    cand_df01 = sample_y2_on_y1(cand_df, 0, 1, py2d, rng)
    cand_df00 = sample_y2_on_y1(cand_df, 0, 0, py2d, rng)

    new_cand_df = cand_df11.append(cand_df10).append(cand_df01).append(cand_df00)
    new_cand_df.reset_index(inplace=True, drop=True)

    # --- Fix the conditional distributions of y1
    cand_df1 = sample_y1_on_main(new_cand_df, 1, py1d, rng)
    cand_df0 = sample_y1_on_main(new_cand_df, 0, py1d, rng)

    cand_df10 = cand_df1.append(cand_df0)
    cand_df10.reset_index(inplace=True, drop=True)

    # --- Fix the marginal
    final_df = fix_marginal(cand_df10, py00, rng)

    return final_df

def save_created_data(df, experiment_directory, filename):
    IMG_MAIN_DIR = "/nfs/turbo/coe-rbg/"
    df['Path'] = IMG_MAIN_DIR + df['Path']
    df.drop(['uid', 'patient', 'study'], axis = 1, inplace = True)
    df.to_csv(f'{experiment_directory}/{filename}.csv', index=False)

def create_save_chexpert_lists(experiment_directory, p_tr=.7, random_seed=None):

	if random_seed is None:
		rng = np.random.RandomState(0)
	else:
		rng = np.random.RandomState(random_seed)

	# --- read in the cleaned image filenames (see chexpert_creation)
	df = pd.read_csv('./penumonia_nofinding_cohort.csv')

	# ---- split into train and test patients
	tr_val_candidates = rng.choice(df.patient.unique(),
		size = int(len(df.patient.unique())*p_tr), replace = False).tolist()
	ts_candidates = list(set(df.patient.unique()) - set(tr_val_candidates))

	# --- split training into training and validation
	# TODO: don't hard code the validation percent
	tr_candidates = rng.choice(tr_val_candidates,
		size=int(0.75 * len(tr_val_candidates)), replace=False).tolist()
	val_candidates = list(set(tr_val_candidates) - set(tr_candidates))

	tr_candidates_df = df[(df.patient.isin(tr_candidates))].reset_index(drop=True)
	val_candidates_df = df[(df.patient.isin(val_candidates))].reset_index(drop=True)
	ts_candidates_df = df[(df.patient.isin(ts_candidates))].reset_index(drop=True)

	# --- checks
	assert len(ts_candidates) + len(tr_candidates) + len(val_candidates) == len(df.patient.unique())
	assert len(set(ts_candidates) & set(tr_candidates)) == 0
	assert len(set(ts_candidates) & set(val_candidates)) == 0
	assert len(set(tr_candidates) & set(val_candidates)) == 0

	# --- get train datasets
	tr_sk_df = get_skewed_data(tr_candidates_df, py1d=0.9, py2d=0.9, py00=0.7, rng=rng)
	save_created_data(tr_sk_df, experiment_directory=experiment_directory,
		filename='skew_train')

	tr_usk_df = get_skewed_data(tr_candidates_df, py1d=0.5, py2d=0.5, py00=0.7, rng=rng)
	save_created_data(tr_usk_df, experiment_directory=experiment_directory,
		filename='unskew_train')

	# --- get validation datasets
	val_sk_df = get_skewed_data(val_candidates_df, py1d=0.9, py2d=0.9, py00=0.7, rng=rng)
	save_created_data(val_sk_df, experiment_directory=experiment_directory,
		filename='skew_valid')

	val_usk_df = get_skewed_data(val_candidates_df, py1d=0.5, py2d=0.5, py00=0.7, rng=rng)
	save_created_data(val_usk_df, experiment_directory=experiment_directory,
		filename='unskew_valid')

	# --- get test
	pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
	for pskew in pskew_list:
		ts_sk_df = get_skewed_data(ts_candidates_df, py1d=pskew, py2d=pskew, py00 = 0.7, rng=rng)
		save_created_data(ts_sk_df, experiment_directory=experiment_directory,
			filename=f'{pskew}_test')

def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)  # Decode a JPEG-encoded image to a uint8 tensor.
	return img

def map_to_image_label(img_dir, label):
    # print("@@@@@@@@@@@@@@@@@@@@@image dir: "+img_dir)
    img = tf.io.read_file(img_dir)
    img = tf.image.decode_jpeg(img, channels=3)

    # Resize and rescale the image
    img_height = 128
    img_width = 128
    
    img = tf.image.resize(img, (img_height, img_width))
    img = img / 255

    # decode number?
    return img, label

def create_dataset(csv_dir, params):
    df = pd.read_csv(csv_dir)

    file_paths = df['Path'].values
    # print(file_paths)
    label = df['Age'].values   # penumonia or not
    label = tf.cast(label, tf.float32)

    batch_size = params['batch_size']
    ds = tf.data.Dataset.from_tensor_slices((file_paths, label))
    # print("!!!!!!!!!!!!!!!!")
    ds = ds.map(map_to_image_label).batch(batch_size)
    return ds

def load_created_data(data_dir, skew_train, params):
    skew_str = 'skew' if skew_train == 'True' else 'unskew'

    train_data = create_dataset(f'{data_dir}/{skew_str}_train.csv', params)
    valid_data = create_dataset(f'{data_dir}/{skew_str}_valid.csv', params)

    test_data_dict = {}
    pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    for pskew in pskew_list:
        test_data = create_dataset(f'{data_dir}/{pskew}_test.csv', params)
        test_data_dict[pskew] = test_data
    
    return train_data, valid_data, test_data_dict

"""
Model
"""
from tensorflow.keras.applications.resnet50 import ResNet50
pre_trained_model = ResNet50(input_shape=(128, 128, 3), 
                             include_top=False, 
                             weights='imagenet')

last_layer = pre_trained_model.get_layer('conv5_block3_3_conv')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output 

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='linear')(x)  
model = Model(pre_trained_model.input, x) 

model.summary()

params = {}
params['batch_size'] = 64
params['epoch_num'] = 20

# STEP 1: Create the dataset csv
# create_save_chexpert_lists(MAIN_DIR + 'age_data')
# print("Finish creating the dataset csv")

# # STEP 2: Create training, validation and testing dataset
data_dir = MAIN_DIR + 'age_data'
train_ds, valid_ds, test_ds_dict = load_created_data(data_dir, False, params)
print("Finish creating the training, validation, and testing dataset")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(train_ds, epochs=params['epoch_num'], validation_data=valid_ds)

model.evaluate(test_ds_dict[0.5])

import matplotlib.pyplot as plt
# %matplotlib inline
# acc = history.history['acc']
# val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))
# fig1 = plt.gcf()
# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# fig1.savefig(accuracy_plot)

# plt.figure()
# plt.show()

fig1 = plt.gcf()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.savefig('age_plot.png')

# plt.figure()
# plt.show()