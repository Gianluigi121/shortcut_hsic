import tensorflow as tf
import numpy as np

"""Measure the accuracy between predictions and age labels within a threshold"""
def accuracy(pred, y, threshold):
    pred = pred.numpy().reshape(pred.shape[0], 1)
    y = y.numpy().reshape(y.shape[0], 1)
    print(f"pred shape: {pred.shape}")
    print(f"y shape: {y.shape}")
    sub = pred-y
    sub = np.abs(sub)
    res = np.ones_like(sub)
    mask = (sub < threshold)
    res[mask] = 0    # Difference within the threshold marked as correct
    print(f"res shape: {res.shape}")
    error_rate = np.sum(res) / res.shape[0]
    return error_rate

class accuracy_metric(tf.keras.metrics.Metric):
  def __init__(self, name='accuracy_metric', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.pos_count = 0
    self.sample_count = 0

  def update_state(self, pred, y, threshold):
    pred = pred.numpy().reshape(pred.shape[0], 1)
    y = y.numpy().reshape(y.shape[0], 1)
    sub = pred - y
    sub = np.abs(sub)
    res = np.ones_like(sub)
    mask = (sub > threshold)
    res[mask] = 0    # Difference over the threshold marked as incorrect
    self.pos_count += np.sum(res)
    self.sample_count += res.shape[0]

  def result(self):
    return self.pos_count / self.sample_count

  def reset_states(self):
    self.pos_count = 0
    self.sample_count = 0