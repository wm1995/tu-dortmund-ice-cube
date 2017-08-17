#!/usr/bin/env python
# SNN

from __future__ import division, print_function

import numpy as np
import argparse

from myTools.data_loader import load_data
from myTools.WaveformGenerator import WaveformGenerator
from myTools.metrics.keras import precision, recall, f1
from myTools.metrics.sklearn import print_metric_results
from myTools.model_tools.model_saver import save_model

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical, Sequence
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def main(
        data=None,
        params={
            'lr': 0.001,
            'conv_dr': 0.2,
            'fc_dr': 0.2,
            'batch_size': 128,
            'no_epochs': 1000,
            'steps_per_epoch': 100,
            'dp_prob': 0.5,
            'batch_norm': False
        },
        no_threads=10,
        implementation=0,
        verbose=True
    ):
    """
    Runs a LSTM recurrent neural network on the waveform data, saving the model to the filestore
    Requires a directory called 'logs' in the same folder for the TensorBoard visualisation
    
    The current neural network structure is:
        lstm > lstm > lstm > softmax

    Arguments:
        data - the Datasets object to run on, if None then loads data (default = None)
        params - a dictionary object containing the following parameters
            lr - the learning rate of the Adam optimiser (default = 0.001)
            conv_dr - the dropout rate for the recurrent state (default = 0.2)
                      (misleadingly named to maintain compatibility with other tools)
            fc_dr - the dropout rate for the alpha dropout layers (default = 0.2)
            no_epochs - the number of epochs to run for
            steps_per_epoch - the number of batches in each epoch
            dp_prob - the proportion of double pulse waveforms shown at train time (default = 0.5)
            batch_norm - unused
            regularise - if true, uses L2 regularisation on the weights for each layer (default = False)
        no_threads - number of threads to use (default is 10, use -1 to set no limit)
        implementation - sets the implementation used by Keras for the LSTM layers (default = 0)
            0 - RNN uses fewer, larger, matrix products (good for CPU but uses more memory)
            1 - Uses fewer, smaller, matrix products (slow on CPU, may be faster than 0 on GPU, uses less memory)
            2 - Combines different gates in LSTM into one matrix (more efficient on GPU)
        verbose - dictates the amount of output that keras gives
    
    No returns

    """
    # Read in data
    if data == None:
        data = load_data(verbose=verbose)

    # Set up CPU, GPU options
    config = None
    if no_threads == -1:
        config = tf.ConfigProto(
                allow_soft_placement=True, 
                device_count = {'CPU': 1, 'GPU': 1}, 
                gpu_options = tf.GPUOptions(allow_growth = True)
            )
    else:
        config = tf.ConfigProto(
                intra_op_parallelism_threads=no_threads, 
                inter_op_parallelism_threads=no_threads,
                allow_soft_placement=True, 
                device_count = {'CPU': 1, 'GPU': 1}, 
                gpu_options = tf.GPUOptions(allow_growth = True)
            )
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Prepare TensorBoard
    tb = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True)

    # Define model
    model = Sequential()

    # Set up regulariser
    regulariser = None
    if params['regularise']:
        regulariser = l2(0.01)

    model.add(LSTM(128,
            input_shape=(128, 1),
            dropout=params['fc_dr'],
            recurrent_dropout=params['conv_dr'],
            kernel_regularizer=regulariser,
            unroll=True,
            return_sequences=True,
            implementation=implementation
        ))
    model.add(LSTM(128,
            dropout=params['fc_dr'],
            recurrent_dropout=params['conv_dr'],
            kernel_regularizer=regulariser,
            unroll=True,
            return_sequences=True,
            implementation=implementation
        ))
    model.add(LSTM(128,
            dropout=params['fc_dr'],
            recurrent_dropout=params['conv_dr'],
            kernel_regularizer=regulariser,
            unroll=True,
            implementation=implementation
        ))
    model.add(Dense(2, activation='softmax'))

    # Set-up optimiser
    optimiser = Adam(lr=params['lr'])

    # Create model
    model.compile(
            optimizer=optimiser, 
            loss='categorical_crossentropy', 
            metrics=['accuracy', precision, recall, f1]
        )

    # Create generators for training, validation
    train_gen = WaveformGenerator(
            data.train, 
            batch_size=params['batch_size'], 
            balanced=True, 
            dp_prob=params['dp_prob']
        )

    val_gen = WaveformGenerator(
            data.val, 
            batch_size=params['batch_size'], 
            balanced=True, 
            dp_prob=params['dp_prob']
        )
        
    # Train model
    model.fit_generator(
            train_gen, 
            steps_per_epoch=params['steps_per_epoch'], 
            epochs=params['no_epochs'], 
            verbose=int(verbose), 
            validation_data=val_gen,
            validation_steps=params['steps_per_epoch'], 
            callbacks=[tb]
        )

    # Save model
    save_model(model, 'lstm', params, verbose=verbose)
    
    # Evaluate model
    test_preds = model.predict(data.val.waveforms, verbose=int(verbose))
    print()
    print_metric_results(data.val.labels, test_preds, data.val.weights, data.val.ids, th=0.5)
    print_metric_results(data.val.labels, test_preds, data.val.weights, data.val.ids, th=0.9)


if __name__ == "__main__":
    # Initialise the arg parser
    parser = argparse.ArgumentParser(
            description="""
            Runs a LSTM recurrent neural network on the waveform data.
            """
        )
    
    # Add arguments
    parser.add_argument(
            '-l', '--learn-rate', 
            help='sets the learning rate for the Adam optimiser',
            type=float, dest='lr', 
            default=1e-3
        )

    parser.add_argument(
            '-d', '--dropout', 
            help='sets the dropout rate for the input layers (default 0.2)',
            type=float, dest='fc_dr', 
            default=0.2
        )

    parser.add_argument(
            '-x', '--recurrent-dropout', 
            help='sets the dropout rate for the recurrent state (default 0.2)',
            type=float, dest='conv_dr', 
            default=0.2
        )

    parser.add_argument(
            '-b', '--batch-size', 
            help='sets the batch size',
            type=int, dest='batch_size', 
            default=128
        )

    parser.add_argument(
            '-e', '--no-epochs', 
            help='sets the number of epochs',
            type=int, dest='no_epochs', 
            default=1e3
        )

    parser.add_argument(
            '-s', '--steps-per-epoch', 
            help='sets the number of batches per epoch',
            type=int, dest='steps_per_epoch', 
            default=100
        )

    parser.add_argument(
            '-p', '--double-pulse-prob', 
            help='sets proportion of double pulse waveforms used at train time (default = 0.5)',
            type=float, dest='dp_prob', 
            default=0.5
        )

    parser.add_argument(
            '-r', '--regularise', 
            help='uses regularisation on each layer',
            action='store_true', dest='regularise', 
            default=False
        )

    parser.add_argument(
            '-i', '--implementation', 
            help='''sets the implementation used by Keras for the LSTM layers (default = 0)
            \t0 - RNN uses fewer, larger, matrix products (good for CPU but uses more memory)
            \t1 - Uses fewer, smaller, matrix products (slow on CPU, may be faster than 0 on GPU, uses less memory)
            \t2 - Combines different gates in LSTM into one matrix (more efficient on GPU)''',
            type=int, dest='implementation',
            default=0 
        )

    parser.add_argument(
            '-t', '--no-threads', 
            help='sets limit on the number of threads to be used (default = 10, if no limit set to -1)',
            type=int, dest='no_threads', 
            default=10
        )

    parser.add_argument(
            '-v', '--verbose', 
            help='sets the verbose mode for the program',
            action='store_true', dest='verbose', 
            default=False
        )

    # Parse the args
    args = parser.parse_args()

    params = {
        'lr': args.lr,
        'conv_dr': args.conv_dr,
        'fc_dr': args.fc_dr,
        'batch_size': args.batch_size,
        'no_epochs': args.no_epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'dp_prob': args.dp_prob,
        'batch_norm': False,
        'regularise': args.regularise
    }

    main(
            params=params, 
            no_threads=args.no_threads, 
            verbose=args.verbose, 
            implementation=args.implementation
        )
