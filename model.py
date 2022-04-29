import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import DenseNet121

def create_architecture(params):
	if (params['architecture'] == 'pretrained_densenet'):
		net = PretrainedDenseNet121(
			embedding_dim=params["embedding_dim"],
			l2_penalty=params["l2_penalty"])
	else:
		raise NotImplementedError(
			"need to implement other architectures")
	return net

class PretrainedDenseNet121(tf.keras.Model):
	"""pretrained Densenet architecture."""
	def __init__(self, embedding_dim=-1, l2_penalty=0.0,
            l2_penalty_last_only=False):
		super(PretrainedDenseNet121, self).__init__()
		self.embedding_dim = embedding_dim

		self.densenet = DenseNet121(include_top=False, weights='imagenet')
		self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

		if not l2_penalty_last_only:
			regularizer = tf.keras.regularizers.l2(l2_penalty)
			for layer in self.densenet.layers:
				if hasattr(layer, 'kernel'):
					self.add_loss(lambda layer=layer: regularizer(layer.kernel))
		
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
		x = self.dense(x)
		return x

# def Densenet121(parmas):
#     pixel = params['pixel']
#     pretrained_model = DenseNet121(input_shape=(pixel, pixel, 3), 
#                              include_top=False, 
#                              weights='imagenet')

#     last_layer = pretrained_model.get_layer('conv5_block16_2_conv')
#     print('last layer output shape: ', last_layer.output_shape)
#     last_output = last_layer.output 

#     x = layers.Flatten()(last_output)
#     x = layers.Dense(1024, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)                  
#     x = layers.Dense(1, activation='linear')(x)  
#     model = Model(pre_trained_model.input, x)
#     return model


# def Resnet50(params):
#     pixel = params['pixel']
#     pretrained_model = ResNet50(input_shape=(pixel, pixel, 3), 
#                              include_top=False, 
#                              weights='imagenet')

#     last_layer = pretrained_model.get_layer('conv5_block3_3_conv')
#     print('last layer output shape: ', last_layer.output_shape)
#     last_output = last_layer.output 

#     x = layers.Flatten()(last_output)
#     # x = layers.Dense(1024, activation='relu')(x)
#     x = layers.Dropout(0.2)(x)                  
#     x = layers.Dense(1, activation='linear')(x)  
#     model = Model(pretrained_model.input, x)
#     return model