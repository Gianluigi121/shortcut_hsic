import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from model import create_architecture
import datetime
import matplotlib.pyplot as plt
import pickle
from collections import OrderedDict
import argparse

from data_builder import create_save_chexpert_lists, load_ds_from_csv
from evaluation import accuracy, accuracy_metric

"""Plot"""
def plot(history, test_loss_dict, test_accuracy_dict, params):
	epoch_num = params['epoch_num']
	l2 = params['l2_penalty']
	# Plot the training loss and validation loss
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(loss))
	plt.figure()
	plt.plot(epochs, loss, '--o', label='Training loss')
	plt.xlabel("Training loss")
	plt.ylabel("Epoch")
	plt.title('Training loss VS Epoch')
	plt.savefig(MAIN_DIR + f'plot/train_loss_epoch{epoch_num}_l2{l2}.png')

	plt.figure()
	plt.plot(epochs, val_loss, '--o', label='Validation loss')
	plt.xlabel("Validation loss")
	plt.ylabel("Epoch")
	plt.title('Validation loss VS Epoch')
	plt.savefig(MAIN_DIR + f'plot/val_loss_epoch{epoch_num}_l2{l2}.png')

	# # Test loss across different skewed datasets
	# plt.figure()
	# plt.plot(list(test_loss_dict.keys()), list(test_loss_dict.values()), "--o")
	# plt.xlabel("skew probability")
	# plt.ylabel("Loss")
	# plt.title("Loss VS Skew Prob(Test dataset)")
	# # plt.show()
	# plt.savefig(MAIN_DIR + f"plot/test_loss_epoch{epoch_num}_l2{l2}.png")

	# # Test accuracy across different skewed datasets
	# plt.figure()
	# plt.plot(list(test_accuracy_dict.keys()), list(test_accuracy_dict.values()), "--o")
	# plt.xlabel("skew probability")
	# plt.ylabel("Accuracy")
	# plt.title("Accuracy VS Skew Prob(Test dataset)")
	# # plt.show()
	# plt.savefig(MAIN_DIR + f"plot/test_accuracy_epoch{epoch_num}_l2{l2}.png")

def main(MAIN_DIR, params):
	model = create_architecture(params)
	# model.summary()
	acc_metric = accuracy_metric(name='ACC')

	# Create training, validation and testing dataset
	data_dir = MAIN_DIR + 'data'
	train_ds, valid_ds, test_ds_dict = load_ds_from_csv(data_dir, False, params)
	print("Finish creating the training, validation, and testing dataset")

	model.compile(optimizer='adam', 
				loss='mean_squared_error', 
				metrics=[acc_metric], 
				run_eagerly=True)
	history = model.fit(train_ds, 
						epochs=params['epoch_num'], 
						validation_data=valid_ds, 
						verbose=1)

	"""Testing"""				
	test_loss_dict = OrderedDict()
	test_accuracy_dict = OrderedDict()
	for skew_prob in test_ds_dict:
		test_ds = test_ds_dict[skew_prob]
		res_list = model.evaluate(test_ds)
		loss = res_list[0]
		acc = res_list[1]
		print(f"Result for Test datset with skew prob: {skew_prob}"
			f'Test Loss: {loss}, '
			f'Test Accuracy: {acc * 100}'
			)
		test_loss_dict[skew_prob] = loss
		test_accuracy_dict[skew_prob] = acc

	loss_file = open('loss.txt', 'wb')
	pickle.dump(test_loss_dict, loss_file)
	loss_file.close()

	accuracy_file = open("accuracy.txt", "wb")
	pickle.dump(test_accuracy_dict, accuracy_file)
	accuracy_file.close()
	plot(history, test_loss_dict, test_accuracy_dict, params)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# optional arguments
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--epoch_num", type=int, default=1)
	parser.add_argument("--pixel", type=int, default=128, help="Image size: pixel*pixel")
	parser.add_argument("--embedding_dim", type=int, default=-1, help="Embedding dim for the feature vector")
	parser.add_argument("--l2_penalty", type=float, default=0.01, help="Regularizer on each layer")
	args = parser.parse_args()

	MAIN_DIR = "/nfs/turbo/coe-rbg/zhengji/age_mask/"
	params = {'batch_size': args.batch_size, 
			  'epoch_num': args.epoch_num,
			  'pixel': args.pixel, 
			  'architecture': 'pretrained_densenet',
			  'embedding_dim': args.embedding_dim,
			  'l2_penalty': args.l2_penalty
			  }
	print(params)
	main(MAIN_DIR, params)