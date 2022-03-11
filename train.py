import tensorflow as tf
from model import Resnet50, Densenet121
from data_builder import create_save_chexpert_lists, load_ds_from_csv
from evaluation import accuracy, accuracy_metric
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm


params = {'batch_size': 32, 
		  'epoch_num': 1,
		  'pixel': 128
         }

MAIN_DIR = "/nfs/turbo/coe-rbg/zhengji/age/"
data_dir = MAIN_DIR + 'data'
train_ds, valid_ds, test_ds_dict = load_ds_from_csv(data_dir, False, params)
curr_model = Resnet50(params)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
train_accuracy = accuracy_metric(name='train_accuracy')
test_accuracy = accuracy_metric(name='test_accuracy')

@tf.function
def train_step(model, x, y):
    # print("train_step")
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_object(pred, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(pred, y)
    train_accuracy(pred, y)

@tf.function
def test_step(model, x, y):
    pred = model(x)
    test_loss = loss_object(pred, y)

    test_loss(pred, y)
    test_accuracy(pred, y)
    
"""Training and Validation"""
@tf.function
def train_valid(params):
    for epoch in range(params['epoch_num']):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

        # for x, y in train_ds:
        #     correct_train_num = train_step(curr_model, x, y)

        for x, y in valid_ds:
            correct_valid_num = test_step(curr_model, x, y)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Train Accuracy: {train_accuracy.result() * 100}, '
            f'Valid Loss: {test_loss.result()}, '
            f'Valid Accuracy: {test_accuracy.result() * 100}'
        )

tf.config.run_functions_eagerly(True)
train_valid(params)
print(tf.__version__)

"""Testing"""
@tf.function
def test():
    test_loss_dict = {}
    test_accuracy_dict = {}
    for skew_prob in test_ds_dict:
        test_loss.reset_states()
        test_accuracy.reset_states()

        test_ds = test_ds_dict[skew_prob]
        for x, y in test_ds:
            test_step(curr_model, x, y)
        print(f"Result for Test datset with skew prob: {skew_prob}"
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
            )
        test_loss_dict[skew_prob] = test_loss.result()
        test_accuracy_dict[skew_prob] = test_accuracy.result()
    test_loss_dict = sorted(test_loss_dict.items())
    test_accuracy_dict = sorted(test_accuracy_dict.items())
    return test_loss_dict, test_accuracy_dict

test_loss_dict, test_accuracy_dict = test()
"""Plot"""
# Test loss across different skewed datasets
plt.figure()
plt.plot(test_loss_dict.keys, test_loss_dict.values, "--o")
plt.xlabel("skew probability")
plt.ylabel("Loss")
plt.title("Loss VS Skew Prob(Test dataset)")
plt.savefig(MAIN_DIR + "test_loss.png")

# Test accuracy across different skewed datasets
plt.figure()
plt.plot(test_accuracy_dict.keys, test_accuracy_dict.values, "--o")
plt.xlabel("skew probability")
plt.ylabel("Accuracy")
plt.title("Accuracy VS Skew Prob(Test dataset)")
plt.savefig(MAIN_DIR + "test_accuracy.png")
