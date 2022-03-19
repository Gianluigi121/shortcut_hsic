"""Commonly used neural network architectures."""

# NOTE:see batch norm issues here https://github.com/keras-team/keras/pull/9965

import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50

def create_architecture(params):
	if (params['architecture'] == 'pretrained_densenet'):
		net = PretrainedDenseNet121(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"], 
			n_classes=params['n_classes'])
	elif (params['architecture'] == 'pretrained_resnet'):
		net = PretrainedResNet50(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"], 
			n_classes=params['n_classes'])

	else:
		raise NotImplementedError(
			"need to implement other architectures")
	return net

class PretrainedDenseNet121(tf.keras.Model):
	"""Densenet121 pretrained on imagenet."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False, n_classes=1):
		super(PretrainedDenseNet121, self).__init__()
		self.embedding_dim = embedding_dim

		self.densenet = DenseNet121(include_top=False, weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.densenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(n_classes,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.densenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x



class PretrainedResNet50(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False, n_classes=1):
		super(PretrainedResNet50, self).__init__()
		self.embedding_dim = embedding_dim

		self.resenet = ResNet50(include_top=False, layers=tf.keras.layers,
			weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.resenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))

		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(n_classes,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.resenet(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x

