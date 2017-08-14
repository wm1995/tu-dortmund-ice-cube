#!/usr/bin/env python
# Contains the metrics defined for use with the Keras datasets
from __future__ import absolute_import, division
from keras import backend as K

# From https://github.com/fchollet/keras/issues/5400
# Adapted to actually make it work
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.

    For use on tf.Tensor objects.
    """
    true_positives = K.sum(K.round(y_true[:, 1] * y_pred[:, 1]))
    predicted_positives = K.sum(K.round(y_pred[:, 1]))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# From https://github.com/fchollet/keras/issues/5400
# Adapted to make it work
def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.

    For use on tf.Tensor objects.
    """
    true_positives = K.sum(K.round(y_true[:, 1] * y_pred[:, 1]))
    possible_positives = K.sum(K.round(y_true[:, 1]))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# From https://github.com/fchollet/keras/issues/5400
def f1(y_true, y_pred):
    """F1 metric.

    Computes a batch-wise average of the F1 metric.

    For use on tf.Tensor objects.
    """
    prc = precision(y_true, y_pred)
    rcl = recall(y_true, y_pred)
    return 2*((prc*rcl)/(prc+rcl))
