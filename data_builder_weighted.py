import pandas as pd
import tensorflow as tf
from copy import deepcopy
import numpy as np
import os

MAIN_DIR = '/nfs/turbo/coe-rbg/zhengji/age_shortcut/'

def sample_age_range(df, y0_value, age_range, p_pen_con_age, rng):
    """
    Select a fix number of samples according to P(pne&age) computed from
    the given P(pen|age)
    """
    age_df = df[(df.Age >= age_range[0]) & (df.Age < age_range[1])]
    p_age = len(age_df) / len(df)   # P(age in range)

    # p(pne&age)= P(pne|age)P(age)
    p_pen_and_age = p_pen_con_age*p_age

    chosen_df = df[(df.y0==y0_value) & (df.Age>=age_range[0]) & (df.Age<age_range[1])]
    chosen_df = chosen_df.reset_index(drop=True)
    num_samples = int(p_pen_and_age*len(chosen_df))
    age1 = age_range[0]
    age2 = age_range[1]
    print(f"y0:{y0_value}, y2 range: [{age1}, {age2}]: {num_samples}")
    chosen_ids = rng.choice(chosen_df.index, size = num_samples)
    
    df_new = chosen_df.iloc[chosen_ids]
    return df_new

def fix_marginal(df, y0_prob, rng):
	y0_group = df.index[(df.y0 == 0)] # y0=0
	y1_group = df.index[(df.y0 == 1)] # y0=1
	
	y1_prob = 1 - y0_prob 
	# y0 group is too small, y1 group is too large, make y1 group smaller
	# y0 group smaller than p(y0) * all data
	if len(y0_group) < (y0_prob/y1_prob) * len(y1_group):
		y0_ids = deepcopy(y0_group).tolist()
		# change y1 group size to P(y1)*len(data)
		y1_ids = rng.choice(
			y1_group, size = int((y1_prob/y0_prob) * len(y0_group)),
			replace = False).tolist()

	# y1 group is too small, y0 group is too large, make y0 group smaller
	# y1 group smaller than p(y1) * all data
	elif len(y1_group) < (y1_prob/y0_prob) * len(y0_group):
		y1_ids = deepcopy(y1_group).tolist()
		# change y0 group size to P(y0)*len(data)
		y0_ids = rng.choice(
			y0_group, size = int((y0_prob/y1_prob)*len(y1_group)), 
			replace = False
		).tolist()

	dff = df.iloc[y1_ids + y0_ids]
	dff.reset_index(inplace = True, drop=True)
	reshuffled_ids = rng.choice(dff.index, size = len(dff.index), replace=False).tolist()
	dff = dff.iloc[reshuffled_ids].reset_index(drop = True)
	return dff

def get_skewed_data(df, age_range_list=[0, 31, 41, 51, 61, 71, 81, 91], 
                    pen1_prob_list=[0.25, 0.35, 0.45, 0.65, 0.75, 0.85, 0.95], y0_prob=0.7, rng=None):
    """
    Return a new dataset where each subgroup has been modified to fit a fixed
    distribution P(pen|age)
    """
    if rng is None:
        rng = np.random.RandomState(0)

    final_df = pd.DataFrame(columns=df.columns)
    for i in range(len(age_range_list)-1):
        age_pair = [age_range_list[i], age_range_list[i+1]]
        pen1_prob = pen1_prob_list[i]
        df_pen1 = sample_age_range(df, y0_value=1, age_range=age_pair, 
                                   p_pen_con_age=pen1_prob, rng=rng)
        df_pen0 = sample_age_range(df, y0_value=0, age_range=age_pair, 
                                   p_pen_con_age=1-pen1_prob, rng=rng)
        final_df = final_df.append(df_pen1).append(df_pen0)
    final_df = final_df.reset_index(drop=True)

    # Fix the marginal distribution
    final_df = fix_marginal(final_df, y0_prob, rng)
    return final_df

def save_created_data(df, experiment_directory, filename):
    df.to_csv(f'{experiment_directory}/{filename}.csv',
        index=False)

# p_train: Probability of the training and validation dataset
def create_save_chexpert_lists(experiment_directory, p_train=0.9, random_seed=None):
    if random_seed is None:
        rng = np.random.RandomState(0)
    else:
        rng = np.random.RandomState(random_seed)

    if not os.path.isdir(experiment_directory):
        os.mkdir(experiment_directory)

    # --- read in the cleaned image filenames (see chexpert_creation)
    df = pd.read_csv(MAIN_DIR+'final_weighted.csv')

    # ---- split into train and test patients
    tr_val_candidates = rng.choice(df.patient.unique(),
        size = int(len(df.patient.unique())*p_train), replace = False).tolist()
    ts_candidates = list(set(df.patient.unique()) - set(tr_val_candidates))

    # --- split training into training and validation
    tr_candidates = rng.choice(tr_val_candidates,
        size=int(0.8 * len(tr_val_candidates)), replace=False).tolist()
    val_candidates = list(set(tr_val_candidates) - set(tr_candidates))

    tr_candidates_df = df[(df.patient.isin(tr_candidates))].reset_index(drop=True)
    val_candidates_df = df[(df.patient.isin(val_candidates))].reset_index(drop=True)
    ts_candidates_df = df[(df.patient.isin(ts_candidates))].reset_index(drop=True)

    # --- checks
    assert len(ts_candidates) + len(tr_candidates) + len(val_candidates) == len(df.patient.unique())
    assert len(set(ts_candidates) & set(tr_candidates)) == 0
    assert len(set(ts_candidates) & set(val_candidates)) == 0
    assert len(set(tr_candidates) & set(val_candidates)) == 0

    age_range_list=[0, 31, 51, 61, 71, 91]
    pen1_prob_list=[0.15, 0.25, 0.85, 0.9, 0.95]
    y0_prob=0.7

    # --- get train datasets
    print("Training Dataset")
    tr_sk_df = get_skewed_data(tr_candidates_df, age_range_list, pen1_prob_list, y0_prob, rng)
    print(f"Training sample number: {len(tr_sk_df)}")
    save_created_data(tr_sk_df, experiment_directory=experiment_directory,
        filename='skew_train')

    # --- get validation datasets
    print("Valid Dataset")
    val_sk_df = get_skewed_data(val_candidates_df, age_range_list, pen1_prob_list, y0_prob, rng)
    print(f"Validation sample number: {len(val_sk_df)}")
    save_created_data(val_sk_df, experiment_directory=experiment_directory,
        filename='skew_valid')

    pen1_prob_list_bal = [0.5]*len(pen1_prob_list)
    pen1_prob_list_unbal = pen1_prob_list
    print("Test Dataset")
    ts_sk_df = get_skewed_data(ts_candidates_df, age_range_list, pen1_prob_list_bal, y0_prob, rng)
    ts_unsk_df = get_skewed_data(ts_candidates_df, age_range_list, pen1_prob_list_unbal, y0_prob, rng)
    print(f"Balanced test sample number: {len(ts_sk_df)}")
    print(f"Unbalanced test sample number: {len(ts_unsk_df)}")
    save_created_data(ts_sk_df, experiment_directory=experiment_directory,
        filename='unskew_test')
    save_created_data(ts_unsk_df, experiment_directory=experiment_directory,
        filename='skew_test')

def map_to_image_label(img_dir, labels):
    # read in a 128*128 image
    img = tf.io.read_file(img_dir)
    img = tf.image.decode_png(img, channels=3)
    img = img / 255

    return img, labels

# Create the dataset with weighted data
def create_dataset(csv_dir, params, is_train):
    df = pd.read_csv(csv_dir)
    df.masked_imgs = '/nfs/turbo/coe-rbg/zhengji/age_shortcut/' + df.masked_imgs
    file_paths = df['masked_imgs'].values
    age = df['Age'].values
    age = tf.cast(age, tf.float32)
    y0 = df['y0'].values
    weight = df['weight'].values
    labels = tf.stack([y0, age, weight], axis=1)

    batch_size = params['batch_size']
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if is_train:
        ds = ds.map(map_to_image_label).shuffle(int(1e5)).batch(batch_size).repeat(10)
    else:
        ds = ds.map(map_to_image_label).shuffle(int(1e5)).batch(batch_size)
    return ds

def load_ds_from_csv(experiment_directory, skew_train, params):
    skew_str = 'skew' if skew_train == 'True' else 'unskew'
    print(skew_str)

    train_csv_dir = f'{experiment_directory}/{skew_str}_train.csv'
    train_ds = create_dataset(train_csv_dir, params, True)

    valid_csv_dir = f'{experiment_directory}/{skew_str}_valid.csv'
    valid_ds = create_dataset(valid_csv_dir, params, False)

    unskew_test_csv_dir = f'{experiment_directory}/unskew_test.csv'
    unskew_test_ds = create_dataset(unskew_test_csv_dir, params, False)

    skew_test_csv_dir = f'{experiment_directory}/skew_test.csv'
    skew_test_ds = create_dataset(skew_test_csv_dir, params, False)

    return train_ds, valid_ds, unskew_test_ds, skew_test_ds

# create_save_chexpert_lists(MAIN_DIR + 'weighted_data')
# params = {'batch_size': 32, 'pixel':128}
# train_ds, valid_ds, unskew_test_ds, skew_test_ds = load_ds_from_csv(MAIN_DIR+'weighted_data', 'True', params)

# for x, y in train_ds.take(2):
#     print(y)