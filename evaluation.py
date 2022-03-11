import tensorflow as tf
import numpy as np

"""Measure the accuracy between predictions and age labels within a threshold"""
def accuracy(pred, y):
    pred = pred.numpy().reshape(pred.shape[0], 1)
    y = y.numpy().reshape(y.shape[0], 1)
    sub = pred-y
    sub = np.abs(sub)
    res = np.ones_like(sub)
    mask = (sub > 2)
    res[mask] = 0    # Difference within the threshold marked as correct
    accuracy = np.sum(res) / res.shape[0]
    return accuracy

class accuracy_metric(tf.keras.metrics.Metric):
  def __init__(self, name='accuracy_metric'):
    super(accuracy_metric, self).__init__(name=name)
    self.pos_count = self.add_weight(name='pos_count', initializer='zeros')
    self.sample_count = self.add_weight(name='sample_count', initializer='zeros')

  def update_state(self, pred, y, sample_weight=None):
    pred = pred.numpy().reshape(pred.shape[0], 1)
    y = y.numpy().reshape(y.shape[0], 1)
    sub = pred - y
    sub = np.abs(sub)
    res = np.ones_like(sub)
    mask = (sub > 2)
    res[mask] = 0    # Difference over the threshold marked as incorrect
    self.pos_count.assign_add(np.sum(res)) 
    self.sample_count.assign_add(res.shape[0])

  def result(self):
    return self.pos_count / self.sample_count

  def reset_states(self):
    self.pos_count.assign(0)
    self.sample_count.assign(0)