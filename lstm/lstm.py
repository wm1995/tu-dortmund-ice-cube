#!/usr/bin/env python
# LSTM

from __future__ import division, print_function

from myTools.seed_setter import check_seed_set

check_seed_set()

import numpy as np
import argparse

from myTools.train_tools.resource_limiter import limit_resources
from myTools.waveform_tools.data_loader import load_data
from myTools.waveform_tools.waveform_generator import WaveformGenerator
from myTools.metrics.keras import precision, recall, f1, class_balance
from myTools.metrics.sklearn import print_metric_results
from myTools.model_tools.model_saver import ModelSaver
from mytools.train_tools.model_trainer import train_model

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape
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
            'batch_norm': False,
            'regularise': 0.0,
            'decay': 0.0
        },
        no_threads=10,
        implementation=0,
        verbose=True,
        cp_interval=100,
        test=False
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
            regularise - sets the amount of L2 regularisation for each layer (default = 0.0)
            decay - sets the decay rate for the proportion of double-pulse waveforms used for 
                    training and validation (default = 0.0)
        no_threads - number of threads to use (default is 10, use 0 to set no limit)
        implementation - sets the implementation used by Keras for the LSTM layers (default = 0)
            0 - RNN uses fewer, larger, matrix products (good for CPU but uses more memory)
            1 - Uses fewer, smaller, matrix products (slow on CPU, may be faster than 0 on GPU, uses less memory)
            2 - Combines different gates in LSTM into one matrix (more efficient on GPU)
            NB in practice, 0 seems to be significantly faster, even with the GPU
        verbose - dictates the amount of output that keras gives
        cp_interval - the number of epochs between saving model checkpoints (default = 100)
        test - suppresses saving of model and output of logs (for testing new features; default = False)
    
    No returns

    """
    # Read in data
    if data == None:
        data = load_data(verbose=verbose)

    limit_resources(no_threads=no_threads)

    # Define model
    model = Sequential()

    # Set up regulariser
    regulariser = l2(params['regularise'])

    # Reshape input to fit with LSTM
    model.add(Reshape((128, 1), input_shape = (128,)))

    # Define model
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
            metrics=['accuracy', precision, recall, f1, class_balance]
        )

    train_model(
            model, data,
            params, 'lstm',
            verbose=verbose
        )
    
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
            '-r', '--regularisation', 
            help='sets amount of regularisation on each layer (default = 0.0)',
            type=float, dest='regularise', 
            default=0.0
        )

    parser.add_argument(
            '-i', '--implementation', 
            help='''sets the implementation used by Keras for the LSTM layers (default = 0)
            \t0 - RNN uses fewer, larger, matrix products (good for CPU but uses more memory)
            \t1 - Uses fewer, smaller, matrix products (slow on CPU, may be faster than 0 on GPU, uses less memory)
            \t2 - Combines different gates in LSTM into one matrix (more efficient on GPU)
            \tNB In practice, 0 seems to be the fastest, even on GPU''',
            type=int, dest='implementation',
            default=0 
        )

    parser.add_argument(
            '-t', '--no-threads', 
            help='sets limit on the number of threads to be used (default = 10, if no limit set to 0)',
            type=int, dest='no_threads', 
            default=10
        )

    parser.add_argument(
            '-v', '--verbose', 
            help='sets the verbose mode for the program',
            action='store_true', dest='verbose', 
            default=False
        )

    parser.add_argument(
            '-k', '--cp-interval', 
            help='sets number of epochs between the saving of model checkpoints',
            type=int, dest='cp_interval', 
            default=100
        )

    parser.add_argument(
            '-D', '--decay', 
            help='sets decay rate for the proportion of double-pulse waveforms used for training and validation (default = 0.0)',
            type=float, dest='decay', 
            default=0.0
        )

    parser.add_argument(
            '--test', 
            help='suppresses saving of model and outputting of logs (use for testing new features)',
            action='store_true', dest='test', 
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
        'regularise': args.regularise,
        'decay': args.decay
    }

    main(
            params=params, 
            no_threads=args.no_threads, 
            verbose=args.verbose, 
            implementation=args.implementation, 
            cp_interval=args.cp_interval,
            test=args.test
        )
