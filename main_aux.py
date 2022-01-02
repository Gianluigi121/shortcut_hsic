import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

import os, shutil
import functools
import datetime
from copy import deepcopy

MAIN_DIR = "/nfs/turbo/coe-rbg/zhengji/single_shortcut/chexpert/"

""" Dataset """
# I. Data Selection: select the data based on the given probability
path = MAIN_DIR + 'data'
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
	df = pd.read_csv(MAIN_DIR + 'penumonia_nofinding_cohort.csv')

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
	tr_sk_df = get_skewed_data(tr_candidates_df, py1d = 0.9, py2d=0.9, py00 = 0.7, rng=rng)
	save_created_data(tr_sk_df, experiment_directory=experiment_directory,
		filename='skew_train')

	tr_usk_df = get_skewed_data(tr_candidates_df, py1d = 0.5, py2d=0.5, py00 = 0.7, rng=rng)
	save_created_data(tr_usk_df, experiment_directory=experiment_directory,
		filename='unskew_train')

	# --- get validation datasets
	val_sk_df = get_skewed_data(val_candidates_df, py1d = 0.9, py2d=0.9, py00 = 0.7, rng=rng)
	save_created_data(val_sk_df, experiment_directory=experiment_directory,
		filename='skew_valid')

	val_usk_df = get_skewed_data(val_candidates_df, py1d = 0.5, py2d=0.5, py00 = 0.7, rng=rng)
	save_created_data(val_usk_df, experiment_directory=experiment_directory,
		filename='unskew_valid')

	# --- get test
	pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
	for pskew in pskew_list:
		ts_sk_df = get_skewed_data(ts_candidates_df, py1d=pskew, py2d=pskew, py00 = 0.7, rng=rng)
		save_created_data(ts_sk_df, experiment_directory=experiment_directory,
			filename=f'{pskew}_test')

# II. Dataset preparation
def read_decode_jpg(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.dtypes.cast(img, tf.float64)
    return img

def decode_number(label):
	label = tf.expand_dims(label, 0)
	label = tf.strings.to_number(label)
	return label

def map_to_image_label(img_dir, label):
    img = read_decode_jpg(img_dir)

    # Resize and rescale the image
    img_height = 128
    img_width = 128
    
    img = tf.image.resize(img, (img_height, img_width))
    img = img / 255
    print(f"Resized image shape: {img.shape}")

    # decode number?
    print(label.shape)
    return img, label

def create_dataset(csv_dir, params):
    df = pd.read_csv(csv_dir)
    
    file_paths = df['Path'].values
    y0 = df['y0'].values   # penumonia or not
    y1 = df['y1'].values   # sex male = 1, female = 0
    y2 = df['y2'].values   # Age >= 50 1, < 50 = 0
    labels = tf.stack([y0, y1, y2], axis = 1)
    print(f"labels shape {labels.shape}")
    labels = tf.cast(labels, tf.float32)

    batch_size = params['batch_size']
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
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

""" Model and Loss"""
# Define the model that return the representation vector of x and the 
from tensorflow.keras.applications.resnet50 import ResNet50

# Define the model
class PretrainedResNet50(tf.keras.Model):
	def __init__(self, embedding_dim=10, l2_penalty=0.0, l2_penalty_last_only=False):
		super(PretrainedResNet50, self).__init__()
		self.embedding_dim = embedding_dim
		self.resnet = ResNet50(include_top=False, layers=tf.keras.layers,
		                       weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
  
		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resnet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		
		if self.embedding_dim != 10:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
   
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs):
		x = self.resnet(inputs)
		x = self.avg_pool(x)
		x = self.embedding(x)
		return self.dense(x), x

def hsic(x, y, sigma=1.0):
    """ Computes the HSIC between two arbitrary variables x, y for kernels with lengthscale sigma"""
    kernel_fxx = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=1.0, length_scale=sigma)
   
    kernel_xx = kernel_fxx.matrix(x, x)
    kernel_fyy = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=1.0, length_scale=sigma)
    kernel_yy = kernel_fyy.matrix(y, y)

    tK = kernel_xx - tf.linalg.diag_part(kernel_xx)
    tL = kernel_yy - tf.linalg.diag_part(kernel_yy)

    N = y.shape[0]
   
    hsic_term1 = tf.linalg.trace(tK @ tL)
    hsic_term2 = tf.reduce_sum(tK) * tf.reduce_sum(tL) / (N -1) / (N - 2)
    hsic_term3 = tf.tensordot(tf.reduce_sum(tK, 0), tf.reduce_sum(tL, 0), 1) / (N - 2)

    return (hsic_term1 + hsic_term2 - 2*hsic_term3) / (N * (N - 3))

def compute_loss_unweighted(labels, logits, z_pred, params):
    # labels: ground truth labels([y0(pnemounia), y1(sex), y2(age)])
    # logits: predicted label(pnemounia)
    # z_pred: a learned representation vector
    y_main = tf.expand_dims(labels[:, 0], axis = -1)

    individual_losses = tf.keras.losses.binary_crossentropy(y_main, logits, from_logits=True)

    unweighted_loss = tf.reduce_mean(individual_losses)
    aux_y = labels[:, 1:]
    hsic_loss = hsic(z_pred, aux_y, sigma = params['sigma'])
    return unweighted_loss, hsic_loss

""" Training """

# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.Accuracy('train_accuracy')
train_auroc = tf.keras.metrics.AUC(name='train_auroc')
eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
eval_accuracy = tf.keras.metrics.Accuracy('eval_accuracy')
eval_auroc = tf.keras.metrics.AUC(name='eval_auroc')
test_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.Accuracy('eval_accuracy')
test_auroc = tf.keras.metrics.AUC(name='eval_auroc')

def auroc(auc_metric, labels, predictions):
	""" Computes AUROC """
	# auc_metric = tf.keras.metrics.AUC(name="auroc")
	# auc_metric.reset_states()
	auc_metric.update_state(y_true=labels, y_pred=predictions)
	return auc_metric.result()

def accuracy(acc, labels, predictions):
    """ Computes Accuracy"""
    # acc = tf.keras.metrics.Accuracy()
    # acc.reset_states()
    acc.update_state(y_true=labels, y_pred=predictions)
    return acc.result()

def update_eval_metrics_dict(acc, acc_metric, labels, predictions):
	y_main = tf.expand_dims(labels[:, 0], axis=-1)

	eval_metrics_dict = {}

	eval_metrics_dict['accuracy'] = accuracy(acc,
		labels=y_main, predictions=predictions["classes"])

	eval_metrics_dict["auc"] = auroc(acc_metric, 
		labels=y_main, predictions=predictions["probabilities"])

	return eval_metrics_dict

# Training function for one step
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits, zpred = model(x)
        # y_pred: predicted probability
        ypred = tf.nn.sigmoid(logits)
        predictions = {
            "classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
            "logits": logits,
            "probabilities": ypred,
            "embedding": zpred
        }
        prediction_loss, hsic_loss = compute_loss_unweighted(y, logits, zpred, params)
        regularization_loss = tf.reduce_sum(model.losses)
        loss = regularization_loss + prediction_loss + params["alpha"] * hsic_loss

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # metrics: accuracy(class), auc(probability)
    train_loss(loss)
    metrics = update_eval_metrics_dict(train_accuracy, train_auroc, y, predictions)
    # print(f"Training accuracy: {metrics['accuracy']}")
    # print(f"Training auc: {metrics['auc']}")

def eval_step(model, x, y):
    # logits: prediction, zpred: representation vec
    logits, zpred = model(x)
    # y_pred: predicted probability
    ypred = tf.nn.sigmoid(logits)
    predictions = {
        "classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
        "logits": logits,
        "probabilities": ypred,
        "embedding": zpred
    }

    # loss
    eval_pred_loss, eval_hsic_loss = compute_loss_unweighted(y, logits, zpred, params)
    loss = (eval_pred_loss + params["alpha"] * eval_hsic_loss).numpy()
    eval_loss(loss)
    
    # metrics
    metrics = update_eval_metrics_dict(eval_accuracy, eval_auroc, y, predictions)
    # print(f"Evaluation accuracy: {metrics['accuracy']}")
    # print(f"Evaluation auroc: {metrics['auc']}")

# Set up summary writers to write the summaries to disk in a different logs directory
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
eval_log_dir = 'logs/gradient_tape/' + current_time + '/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

def train_eval(params):
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    model = PretrainedResNet50(embedding_dim=params['embedding_dim'], l2_penalty=params['l2_penalty'])

    for epoch in range(params['num_epochs']):
        print(f"\nTraining epoch {epoch}")
        for step, (x, y) in enumerate(train_ds):
            train_step(model, optimizer, x, y)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('auroc', train_auroc.result(), step=epoch)
        print("\n------------------------------")
        print(f"Training result for epoch {epoch}")
        print(f"loss: {train_loss.result()}")
        print(f"accuracy: {train_accuracy.result()}")
        print(f"auroc: {train_auroc.result()}")
    
        for step, (x, y) in enumerate(valid_ds):
            eval_step(model, x, y)
        with eval_summary_writer.as_default():
            tf.summary.scalar('loss', eval_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', eval_accuracy.result(), step=epoch)
            tf.summary.scalar('auroc', eval_auroc.result(), step=epoch)
        
        print("\n------------------------------")
        print(f"Evaluation result for epoch {epoch}")
        print(f"loss: {eval_loss.result()}")
        print(f"accuracy: {eval_accuracy.result()}")
        print(f"auroc: {eval_auroc.result()}")
        
        # Reset metrics every epoch
        train_loss.reset_states()
        eval_loss.reset_states()
        train_accuracy.reset_states()
        eval_accuracy.reset_states()
        train_auroc.reset_states()
        eval_auroc.reset_states()

    # Save and return the trained model
    # epoch_num = params['num_epochs']
    model.save_weights(MAIN_DIR + 'weights.h5')
    return model

def test_step(model, x, y):
    # logits: prediction, zpred: representation vec
    logits, zpred = model(x)
    # y_pred: predicted probability
    ypred = tf.nn.sigmoid(logits)
    predictions = {
        "classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
        "logits": logits,
        "probabilities": ypred,
        "embedding": zpred
    }

    # loss
    test_pred_loss, test_hsic_loss = compute_loss_unweighted(y, logits, zpred, params)
    loss = (test_pred_loss + params["alpha"] * test_hsic_loss).numpy()
    test_loss(loss)
    
    # metrics
    metrics = update_eval_metrics_dict(test_accuracy, test_auroc, y, predictions)
    # print(f"Testing accuracy: {metrics['accuracy']}")
    # print(f"Testing auc: {metrics['auc']}")

def test(model, params):
    # model = tf.keras.models.load_model(model_dir)
    
    pskew_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    for pskew in pskew_list:
        test_ds = test_ds_dict[pskew]
        for step, (x, y) in enumerate(test_ds):
            test_step(model, x, y)
        print("\n*****************************")
        print(f"Test result for pskew={pskew}")
        print(f"loss: {test_loss.result()}")
        print(f"accuracy: {test_accuracy.result()}")
        print(f"auroc: {test_auroc.result()}")

# Define params dictionary
params = {}
params['embedding_dim'] = 1000
params['l2_penalty'] = 0.0
params['num_epochs'] = 20
params['alpha'] = 1.0  # parameter for HSIC loss
params['batch_size'] = 64
params['sigma'] = 1.0
params['lr'] = 1e-5

# STEP 1: Create the dataset csv
# create_save_chexpert_lists(MAIN_DIR+'data')
# print("Finish creating the dataset csv")

# STEP 2: Create training, validation and testing dataset
data_dir = MAIN_DIR + 'data'
train_ds, valid_ds, test_ds_dict = load_created_data(data_dir, True, params)
# train_ds = create_dataset(MAIN_DIR + 'data/unskew_train.csv', params)
# valid_ds = create_dataset(MAIN_DIR + 'data/unskew_valid.csv', params)
print("Finish creating the training, validation, and testing dataset")

# # STEP 3: Training and evaluating
model = train_eval(params)

# # STEP 4: Testing
# model_dir = MAIN_DIR + 'model.h5'
test(model, params)