""" Script for training """
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_probability as tfp
import datetime


# --- Dataset prep functions
def read_decode_jpg(file_path):
	img = tf.io.read_file(file_path)
	img = tf.image.decode_jpeg(img, channels=3)
	return img


def decode_number(label):
	label = tf.expand_dims(label, 0)
	label = tf.strings.to_number(label)
	return label


def map_to_image_label(img_dir, label):
	img = read_decode_jpg(img_dir)

	# TODO: don't hardcode pixels
	# Resize and rescale the image
	img_height = 128
	img_width = 128

	img = tf.image.resize(img, (img_height, img_width))
	img = img / 255
	return img, label


def create_dataset(csv_dir, params):
	df = pd.read_csv(csv_dir)

	file_paths = df['Path'].values
	y0 = df['y0'].values   # penumonia or not
	y1 = df['y1'].values   # sex male = 1, female = 0
	y2 = df['y2'].values   # support device or not
	labels = tf.stack([y0, y1, y2], axis=1)
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


# ---- model setup



# Define the model
class PretrainedResNet50(tf.keras.Model):
	def __init__(self, embedding_dim=-1, l2_penalty=0.0, l2_penalty_last_only=False):
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

		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs):
		x = self.resnet(inputs)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x

class PretrainedDenseNet121(tf.keras.Model):
	"""pretrained Densenet architecture."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False):
		super(PretrainedDenseNet121, self).__init__()
		self.embedding_dim = embedding_dim

		self.densenet = DenseNet121(include_top=False,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.densenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))

		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(1,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.densenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x


# ---- loss setup
def hsic(x, y, sigma=1.0):
	""" Computes the HSIC between two arbitrary variables x, y"""

	if len(x.shape) == 1:
		x = tf.expand_dims(x, axis=-1)

	if len(y.shape) == 1:
		y = tf.expand_dims(y, axis=-1)

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

	return (hsic_term1 + hsic_term2 - 2 * hsic_term3) / (N * (N - 3))


def compute_loss_unweighted(labels, logits, z_pred, params):
	# labels: ground truth labels([y0(pnemounia), y1(sex), y2(support device)])
	# logits: predicted label(pnemounia)
	# z_pred: a learned representation vector
	y_main = tf.expand_dims(labels[:, 0], axis=-1)

	individual_losses = tf.keras.losses.binary_crossentropy(y_main, logits,
		from_logits=True)

	unweighted_loss = tf.reduce_mean(individual_losses)
	aux_y = labels[:, 1:]
	if params['alpha'] > 0:
		hsic_loss = hsic(z_pred, aux_y, sigma=params['sigma'])
	else:
		hsic_loss = 0.0
	return unweighted_loss, hsic_loss


def accuracy(acc, labels, predictions):
	""" Computes Accuracy"""
	# acc = tf.keras.metrics.Accuracy()
	# acc.reset_states()
	acc.update_state(y_true=labels, y_pred=predictions)
	return acc.result()

def auroc(auc_metric, labels, predictions):
	""" Computes AUROC """
	# auc_metric = tf.keras.metrics.AUC(name="auroc")
	# auc_metric.reset_states()
	auc_metric.update_state(y_true=labels, y_pred=predictions)
	return auc_metric.result()

def update_eval_metrics_dict(acc, acc_metric, labels, predictions):
	y_main = tf.expand_dims(labels[:, 0], axis=-1)

	eval_metrics_dict = {}

	eval_metrics_dict['accuracy'] = accuracy(acc,
		labels=y_main, predictions=predictions["classes"])

	eval_metrics_dict["auc"] = auroc(acc_metric,
		labels=y_main, predictions=predictions["probabilities"])

	return eval_metrics_dict

# ---- training and evaluation
def create_metric_dict(group):
	group_loss = tf.keras.metrics.Mean(f'{group}_loss', dtype=tf.float32)
	group_accuracy = tf.keras.metrics.Accuracy(f'{group}_accuracy')
	group_auroc = tf.keras.metrics.AUC(name=f'{group}_auroc')

	return {'loss': group_loss, 'accuracy': group_accuracy, 'auroc': group_auroc}


def train_step(model, optimizer, x, y, params, metric_dict):
	train_loss = metric_dict['loss']
	train_accuracy = metric_dict['accuracy']
	train_auroc = metric_dict['auroc']

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


def eval_step(model, x, y, params, metric_dict):
	eval_loss = metric_dict['loss']
	eval_accuracy = metric_dict['accuracy']
	eval_auroc = metric_dict['auroc']

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


def train_eval(params, train_ds, valid_ds, save_directory, scratch_directory):

	if not os.path.exists(scratch_directory):
		os.makedirs(scratch_directory)

	if not os.path.exists(save_directory):
		os.makedirs(save_directory)

	# Set up summary writers to write the summaries to disk
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	train_log_dir = f'{scratch_directory}/' + current_time + '/train'
	eval_log_dir = f'{scratch_directory}/' + current_time + '/eval'
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)

	# create metric dictionaries
	train_metric_dict = create_metric_dict('train')
	eval_metric_dict = create_metric_dict('eval')

	# set up the optimizer
	optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
	model = PretrainedResNet50(embedding_dim=params['embedding_dim'],
		l2_penalty=params['l2_penalty'])

	# start the training/eval
	for epoch in range(params['num_epochs']):
		print(f"\nTraining epoch {epoch}")
		for step, (x, y) in enumerate(train_ds):
			print(f"Step = {step}")
			train_step(model, optimizer, x, y, params, train_metric_dict)
		with train_summary_writer.as_default():
			tf.summary.scalar('loss', train_metric_dict['loss'].result(), step=epoch)
			tf.summary.scalar('accuracy', train_metric_dict['accuracy'].result(),
				step=epoch)
			tf.summary.scalar('auroc', train_metric_dict['auroc'].result(), step=epoch)
		print("\n------------------------------")
		print(f"Training result for epoch {epoch}")
		print(f"loss: {train_metric_dict['loss'].result()}")
		print(f"accuracy: {train_metric_dict['accuracy'].result()}")
		print(f"auroc: {train_metric_dict['auroc'].result()}")

		for step, (x, y) in enumerate(valid_ds):
			eval_step(model, x, y, params, eval_metric_dict)
		with eval_summary_writer.as_default():
			tf.summary.scalar('loss', eval_metric_dict['loss'].result(), step=epoch)
			tf.summary.scalar('accuracy', eval_metric_dict['accuracy'].result(), step=epoch)
			tf.summary.scalar('auroc', eval_metric_dict['auroc'].result(), step=epoch)

		print("\n------------------------------")
		print(f"Evaluation result for epoch {epoch}")
		print(f"loss: {eval_metric_dict['loss'].result()}")
		print(f"accuracy: {eval_metric_dict['accuracy'].result()}")
		print(f"auroc: {eval_metric_dict['auroc'].result()}")

		# Reset metrics every epoch
		train_metric_dict['loss'].reset_states()
		train_metric_dict['accuracy'].reset_states()
		train_metric_dict['auroc'].reset_states()

		eval_metric_dict['loss'].reset_states()
		eval_metric_dict['accuracy'].reset_states()
		eval_metric_dict['auroc'].reset_states()

	# Save and return the trained model
	# epoch_num = params['num_epochs']
	model.save_weights(save_directory + 'weights.h5')
	return model

def test_step(model, x, y, params, metric_dict):
	test_loss = metric_dict['loss']
	test_accuracy = metric_dict['accuracy']
	test_auroc = metric_dict['auroc']

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


def test(model, test_ds_dict, params):
	# model = tf.keras.models.load_model(model_dir)
	pskew_list = [0.1, 0.5, 0.9]
	for pskew in pskew_list:
		test_ds = test_ds_dict[pskew]
		test_metric_dict = create_metric_dict(f'test{pskew}')

		for _, (x, y) in enumerate(test_ds):
			test_step(model, x, y, params, test_metric_dict)
		print("\n*****************************")
		print(f"Test result for pskew={pskew}")
		print(f"loss: {test_metric_dict['loss'].result()}")
		print(f"accuracy: {test_metric_dict['accuracy'].result()}")
		print(f"auroc: {test_metric_dict['auroc'].result()}")


def main():
	# TODO: automate the creation of the params
	params = {
		'random_seed': 0,
		'batch_size': 64,
		'alpha': 0.0,
		'sigma': 10.0,
		'lr': 0.001,
		'embedding_dim': -1,
		'l2_penalty': 0,
		'num_epochs': 5
	}

	data_dir = '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/experiment_data/rs0'
	save_directory = '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/tuning/'
	scratch_directory = '/nfs/turbo/coe-rbg/mmakar/multiple_shortcut/chexpert/scratch/'

	# --- load the datasets
	train_data, valid_data, test_data_dict = load_created_data(data_dir,
		'True', params)
	# --- start the training
	model = train_eval(params, train_data, valid_data, save_directory,
		scratch_directory)

	# --- evaluation
	test(model, test_ds_dict, params)

if __name__ == "__main__":
	main()