import tensorflow as tf
from model import Resnet50, Densenet121
from data_builder import create_save_chexpert_lists, load_ds_from_csv
from evaluation import accuracy, accuracy_metric

params = {'batch_size': 32, 
		  'epoch_num': 10,
		  'pixel': 128}
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
def train_step(model, x, y, loss_object, train_loss, train_accuracy):
    print("train_step")
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_object(pred, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(pred, y)
    train_accuracy(pred, predictions, threshold=2)

@tf.function
def test_step(model, x, y, loss_object, test_loss, test_accuracy):
    pred = model(x)
    test_loss = loss_object(pred, y)
    test_loss(pred, y)
    test_accuracy(pred, predictions, threshold=2)
    
"""Training and Validation"""
EPOCHS = 1
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

    for x, y in train_ds:
        train_step(curr_model, x, y, loss_object, train_loss, train_accuracy)

    for x, y in valid_ds:
        test_step(curr_model, x, y, loss_object, test_loss, test_accuracy)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Train Accuracy: {train_accuracy.result() * 100}, '
        f'Valid Loss: {test_loss.result()}, '
        f'Valid Accuracy: {test_accuracy.result() * 100}'
    )

"""Testing"""
for skew_prob in test_ds_dict:
    test_ds = test_ds_dict[skew_prob]
    for x, y in test_ds:
        test_step(model, x, y, loss, test_loss)
    print(f"Result for Test datset with skew prob: {skew_prob}"
          f'Test Loss: {test_loss.result()}, '
          f'Test Accuracy: {test_accuracy.result() * 100}')


