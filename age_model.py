import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
import datetime

from data_builder import create_save_chexpert_lists, load_ds_from_csv
from evaluation import accuracy

MAIN_DIR = "/nfs/turbo/coe-rbg/zhengji/age/"

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

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(1, activation='linear')(x)  
model = Model(pre_trained_model.input, x) 
model.summary()

params = {'batch_size': 16, 
		  'epoch_num': 200,
		  'pixel': 128}

# # STEP 1: Create the dataset csv
# create_save_chexpert_lists(MAIN_DIR + 'data')
# print("Finish creating the dataset csv")

# # STEP 2: Create training, validation and testing dataset
data_dir = MAIN_DIR + 'data'
train_ds, valid_ds, test_ds_dict = load_ds_from_csv(data_dir, False, params)
print("Finish creating the training, validation, and testing dataset")

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(train_ds, epochs=params['epoch_num'], validation_data=valid_ds)

test_ds = test_ds_dict[0.5]
for x, y in test_ds.take(2):
	pred = model(x)
	print(f"prediction shape: {pred.shape}")	# should be (batch_size,)
	error_rate = accuracy(pred, y, 2)
	print(f"error rate: {error_rate}")
# model.evaluate(test_ds_dict[0.5])

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