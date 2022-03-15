import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from model import Resnet50, Densenet121
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
	# Plot the training loss and validation loss
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(loss))
	plt.figure()
	plt.plot(epochs, loss, 'r', '--o', label='Training loss')
	plt.plot(epochs, val_loss, 'b', '--o', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend(loc=0)
	plt.savefig(MAIN_DIR + f'plot/age_plot_epoch{epoch_num}.png')


	# Test loss across different skewed datasets
	plt.figure()
	plt.plot(list(test_loss_dict.keys()), list(test_loss_dict.values()), "--o")
	plt.xlabel("skew probability")
	plt.ylabel("Loss")
	plt.title("Loss VS Skew Prob(Test dataset)")
	# plt.show()
	plt.savefig(MAIN_DIR + f"plot/test_loss_epoch{epoch_num}.png")

	# Test accuracy across different skewed datasets
	plt.figure()
	plt.plot(list(test_accuracy_dict.keys()), list(test_accuracy_dict.values()), "--o")
	plt.xlabel("skew probability")
	plt.ylabel("Accuracy")
	plt.title("Accuracy VS Skew Prob(Test dataset)")
	# plt.show()
	plt.savefig(MAIN_DIR + f"plot/test_accuracy_epoch{epoch_num}.png")

def main(MAIN_DIR, params):
	model = Resnet50(params)
	model.summary()
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
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--epoch_num", type=int, default=1)
	parser.add_argument("--pixel", type=int, default=128, help="Image size: pixel*pixel")
	args = parser.parse_args()

	MAIN_DIR = "/nfs/turbo/coe-rbg/zhengji/age/"
	params = {'batch_size': args.batch_size, 
			  'epoch_num': args.epoch_num,
			  'pixel': args.pixel}
	print(params)
	main(MAIN_DIR, params)