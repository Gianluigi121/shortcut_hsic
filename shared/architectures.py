"""Commonly used neural network architectures."""

# NOTE:see batch norm issues here https://github.com/keras-team/keras/pull/9965

import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import initializers


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
	elif (params['architecture'] == 'pretrained_inception'):
		net = PretrainedInceptionv3(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"],
			n_classes=params['n_classes'])
	elif (params['architecture'] == 'simple_conv'):
		net = SimpleConvolutionNet(l2_penalty=params["l2_penalty"])

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


class PretrainedInceptionv3(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
		l2_penalty_last_only=False, n_classes=1):
		super(PretrainedInceptionv3, self).__init__()
		self.embedding_dim = embedding_dim

		self.inception = InceptionV3(include_top=False,
			# weights='imagenet')
			weights=('/data/ddmg/users/mmakar/projects/multiple_shortcut'
				'/shortcut_hsic/shared/inception_v3_weights_tf_dim_ordering'
				'_tf_kernels_notop.h5'))
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.inception.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))

		if self.embedding_dim != -1:
			self.embedding = tf.keras.layers.Dense(self.embedding_dim,
				kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))
		self.dense = tf.keras.layers.Dense(n_classes,
			kernel_regularizer=tf.keras.regularizers.l2(l2_penalty))

	@tf.function
	def call(self, inputs, training=False):
		x = self.inception(inputs, training)
		x = self.avg_pool(x)
		if self.embedding_dim != -1:
			x = self.embedding(x)
		return self.dense(x), x

class SimpleConvolutionNet(tf.keras.Model):
	"""Simple architecture with convolutions + max pooling."""

	def __init__(self, l2_penalty=0.0):
		super(SimpleConvolutionNet, self).__init__()
		# self.scale = preprocessing.Rescaling(1.0 / 255)
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu", kernel_initializer="zeros")
		self.conv2 = tf.keras.layers.Conv2D(64, 3, activation="relu",  kernel_initializer="zeros")
		self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
		# self.dropout = tf.keras.layers.Dropout(dropout_rate)

		self.flatten1 = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(
			1000,
			activation="relu",
			kernel_regularizer=tf.keras.regularizers.L2(l2=l2_penalty),
			kernel_initializer=initializers.Zeros(),
			bias_initializer=initializers.Zeros(),
			name="Z")
		self.dense2 = tf.keras.layers.Dense(1, kernel_initializer=initializers.Zeros(),
			bias_initializer=initializers.Zeros())

	def call(self, inputs, training=False):
		z = self.conv1(inputs)
		z = self.conv2(z)
		z = self.maxpool1(z)
		# if training:
		# 	z = self.dropout(z, training=training)
		z = self.flatten1(z)
		z = self.dense1(z)
		return self.dense2(z), z