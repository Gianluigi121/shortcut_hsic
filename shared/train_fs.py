"""Main training protocol used for training."""
import os
import pickle
import copy
import tensorflow as tf

from shared import architectures
from shared import evaluation
from shared import train_utils

def serving_input_fn():
	"""Serving function to facilitate model saving."""
	# feat = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 28, 28, 3])
	feat = tf.python.ops.array_ops.placeholder(
		dtype=tf.python.framework.dtypes.float32)
	return tf.estimator.export.TensorServingInputReceiver(features=feat,
		receiver_tensors=feat)


def serving_input_fn_simple_arch():
	"""Serving function to facilitate model saving."""
	feat = tf.python.ops.array_ops.placeholder(
		dtype=tf.python.framework.dtypes.float32, shape=[None, 28, 28, 3])
	return tf.estimator.export.TensorServingInputReceiver(features=feat,
		receiver_tensors=feat)


def model_fn(features, labels, mode, params):
	""" Main training function ."""

	net = architectures.create_architecture(params)

	training_state = mode == tf.estimator.ModeKeys.TRAIN
	logits, zpred = net(features, training=training_state)
	ypred = tf.nn.softmax(logits)

	predictions = {
		"classes": tf.cast(tf.math.greater_equal(ypred, .5), dtype=tf.float32),
		"logits": logits,
		"probabilities": ypred,
		"embedding": zpred
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			export_outputs={
				"classify": tf.estimator.export.PredictOutput(predictions)
			})


	labels = labels['labels']

	if mode == tf.estimator.ModeKeys.EVAL:
		main_eval_metrics = {}

		# -- main loss components
		loss = 	tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, logits,
		from_logits=True))

		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, train_op=None)

	if mode == tf.estimator.ModeKeys.TRAIN:
		opt = tf.keras.optimizers.Adam()
		global_step = tf.compat.v1.train.get_global_step()

		ckpt = tf.train.Checkpoint(
			step=global_step, optimizer=opt, net=net)

		with tf.GradientTape() as tape:
			logits, zpred = net(features, training=training_state)
			ypred = tf.nn.softmax(logits)

			prediction_loss = tf.reduce_mean(
				tf.keras.losses.binary_crossentropy(labels, logits,
				from_logits=True))

			regularization_loss = tf.reduce_sum(net.losses)
			loss = regularization_loss + prediction_loss 

		variables = net.trainable_variables
		gradients = tape.gradient(loss, variables)

		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			train_op=tf.group(
				opt.apply_gradients(zip(gradients, variables)),
				ckpt.step.assign_add(1)))


def train(exp_dir,
					checkpoint_dir,
					dataset_builder,
					architecture,
					training_steps,
					pixel,
					n_classes,
					num_epochs,
					batch_size,
					weighted,
					l2_penalty,
					embedding_dim,
					random_seed,
					cleanup,
					debugger):
	"""Trains the estimator."""

	if not os.path.exists(exp_dir):
		print(f'!=! Making directory {exp_dir} !=!')
		os.makedirs(exp_dir)

	if not os.path.exists(checkpoint_dir):
		print(f'!=! Making directory {checkpoint_dir} !=!')
		os.makedirs(checkpoint_dir)

	train_utils.cleanup_directory(checkpoint_dir)

	input_fns = dataset_builder()
	train_data_size, train_input_fn, valid_input_fn, eval_input_fn_creater = input_fns
	steps_per_epoch = int(train_data_size / batch_size)

	params = {
		"pixel": pixel,
		"architecture": architecture,
		"num_epochs": num_epochs,
		"batch_size": batch_size,
		"steps_per_epoch": steps_per_epoch,
		"weighted": weighted,
		"l2_penalty": l2_penalty,
		"embedding_dim": embedding_dim,
		"n_classes": n_classes
	}

	if debugger == 'True':
		save_checkpoints_steps = 50
	else:
		save_checkpoints_steps = 100000

	run_config = tf.estimator.RunConfig(
		tf_random_seed=random_seed,
		save_checkpoints_steps=save_checkpoints_steps,
		# keep_checkpoint_max=2
		)

	est = tf.estimator.Estimator(
		model_fn, model_dir=checkpoint_dir, params=params, config=run_config)
	print(f"=====steps_per_epoch {steps_per_epoch}======")
	if training_steps == 0:
		training_steps = int(params['num_epochs'] * steps_per_epoch)

	print(f'=======TRAINING STEPS {training_steps}=============')
	est.train(train_input_fn, steps=training_steps)

	validation_results = est.evaluate(valid_input_fn)
	results = {"validation": validation_results}
	# save model
	est.export_saved_model(f'{exp_dir}/saved_model', serving_input_fn)

	if ((cleanup == 'True') & (debugger == 'False')):
		train_utils.cleanup_directory(checkpoint_dir)
