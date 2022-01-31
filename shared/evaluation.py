""" Evaluation metrics """
import tensorflow as tf
import tensorflow_probability as tfp


def compute_loss(labels, logits, z_pred, params):
	if params['weighted'] == 'False':
		prediction_loss, hsic_loss = compute_loss_unweighted(labels, logits,
			z_pred, params)
	else:
		raise NotImplementedError("only implemented without weighting")
	return prediction_loss, hsic_loss


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


def auroc(labels, predictions):
	""" Computes AUROC """
	auc_metric = tf.keras.metrics.AUC(name="auroc")
	auc_metric.reset_states()
	auc_metric.update_state(y_true=labels, y_pred=predictions)
	return auc_metric

def get_prediction_by_group(labels, predictions):

	mean_pred_dict = {}

	for y0_val in [0, 1]:
		for y1_val in [0, 1]:
			for y2_val in [0, 1]:
				y0_mask = y0_val * labels[:, 0] + (1.0 - y0_val) * (1.0 - labels[:, 0])
				y1_mask = y1_val * labels[:, 1] + (1.0 - y1_val) * (1.0 - labels[:, 1])
				y2_mask = y2_val * labels[:, 2] + (1.0 - y2_val) * (1.0 - labels[:, 2])

				labels_mask = tf.where(y0_mask * y1_mask * y2_mask)
				mean_pred_dict[f'mean_pred_{y0_val}{y1_val}{y2_val}'] = tf.compat.v1.metrics.mean(
					tf.gather(predictions, labels_mask)
				)

	return mean_pred_dict


def get_eval_metrics_dict(labels, predictions, params):
	del params
	y_main = tf.expand_dims(labels[:, 0], axis=-1)

	eval_metrics_dict = {}
	eval_metrics_dict["auc"] = auroc(
		labels=y_main, predictions=predictions["probabilities"])

	mean_pred_dict = get_prediction_by_group(labels, predictions["probabilities"])

	return {**eval_metrics_dict, **mean_pred_dict}

