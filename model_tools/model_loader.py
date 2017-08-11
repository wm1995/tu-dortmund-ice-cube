#!/usr/bin/env python
# Script to load a pretrained and saved model

# Do we want to start saving metadata about how each model was trained? Could be wise
#

# What basic structure do we want?

# Options:
#  - load a particular model
#  - read the metadata of a model
#  - load data into memory, train on data (do by default)
#  - Spit out parameters about that model
#  - list all models

# Can generate a dictionary with all the pertinent parameters
# Better to generate a csv?
# Maybe better yet to have one master csv file with info about each
# What are the pertinent parameters?
# Batch size, dropout rate (both of them), batches per epoch, number of epochs, learning rate
# Accuracy, precision, recall, f1 score, learning rate, commentary
from __future__ import print_function

import argparse

import tensorflow as tf

from keras.models import load_model
from keras import backend as K

from myTools.data_loader import load_data
from myTools.metrics.keras import precision, recall, f1
from myTools.metrics.sklearn import print_metric_results

# Following code based on https://docs.python.org/2.7/library/argparse.html
parser = argparse.ArgumentParser(description='Load a pretrained Keras model')
parser.add_argument('-m', '--model-file', dest='filename', nargs='?', help='hdf5 file to be loaded')
parser.add_argument('-d', dest='load_data', action='store_true', help='load default dataset into data')
args = parser.parse_args()

# If data is requested, load data
data = None
if args.load_data:
    data = load_data()

# To load model, need to specify the custom objects we loaded
custom_obj_dict = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Set up CPU, GPU options
config = tf.ConfigProto(
        intra_op_parallelism_threads=10, 
        inter_op_parallelism_threads=10,
        allow_soft_placement=True, 
        device_count = {'CPU': 1, 'GPU': 1}, 
        gpu_options = tf.GPUOptions(allow_growth=True)
    )
sess = tf.Session(config=config)
K.set_session(sess)

model = load_model(args.filename, custom_objects=custom_obj_dict)

testPreds = None
if args.load_data:
    testPreds = model.predict(data.val.waveforms, verbose=1)
    print()
    print_metric_results(data.val.labels, testPreds, data.val.weights, data.val.ids, th=0.5)
    print_metric_results(data.val.labels, testPreds, data.val.weights, data.val.ids, th=0.9)

