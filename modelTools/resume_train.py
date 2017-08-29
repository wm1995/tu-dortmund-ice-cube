# Options:
#  - load a particular model
#  - read the metadata of a model
#  - load data into memory, train on data (do by default)
#  - Spit out parameters about that model
#  - list all models
from __future__ import division, print_function

import csv
import argparse

from myTools.data_loader import load_data
from myTools.WaveformGenerator import WaveformGenerator
from myTools.metrics.keras import precision, recall, f1, class_balance
from myTools.metrics.sklearn import print_metric_results
from myTools.model_tools.model_saver import ModelSaver, MODEL_DIR, MODEL_SUMMARY
from myTools.model_tools.model_loader import load_model, load_uncompiled_model

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

def main(
        filepath,
        curr_epoch,
        params,
        data=None,
        no_threads=10,
        verbose=True,
        cp_interval=100,
        test=False
    ):
    """
    Resumes training for the specified Keras model

    Requires a directory called 'logs' in the same folder for the TensorBoard visualisation

    Arguments:
        filepath - the path to the specified HDF5 file containing the Keras model (must be in the model store)
        curr_epoch - the number of the last completed epoch (1-indexed)
        data - the Datasets object to run on, if None then loads data (default = None)
        params - a dictionary object containing the following parameters
            lr - the learning rate of the Adam optimiser (default = 0.001)
            conv_dr - the dropout rate after the convolutional layers (default = 0.7)
            fc_dr - the dropout rate after the fully-connected layers (default = 0.5)
            no_epochs - the number of epochs to run for
            steps_per_epoch - the number of batches in each epoch
            dp_prob - the proportion of double pulse waveforms shown at train time (default = 0.5)
            batch_norm - if true, use batch norm after each layer
            regularise - sets the amount of L2 regularisation for each layer (default = 0.0)
            decay - sets the decay rate for the proportion of double-pulse waveforms used for 
                    training and validation (default = 0.0)
        no_threads - number of threads to use (default is 10, use -1 to set no limit)
        verbose - dictates the amount of output that keras gives
        cp_interval - the number of epochs between saving model checkpoints (default = 100)
        test - suppresses saving of model and output of logs (for testing new features; default = False)
    
    No returns

    """
    model_name = filepath.split('/')[-1]

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

    # Get model
    if params['lr'] != None:
        # Load model
        model = load_uncompiled_model(filepath)

        optimiser = Adam(params['lr'])

        # Create model
        model.compile(
                optimizer=optimiser, 
                loss='categorical_crossentropy', 
                metrics=['accuracy', precision, recall, f1, class_balance]
            )
    else:
        model = load_model(filepath)

    # Read initial parameters
    # Code adapted from https://docs.python.org/2/library/csv.html
    with open(MODEL_DIR + MODEL_SUMMARY) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['model_name'] == model_name:
                old_params = row
                break

    # Compare parameters
    for key in old_params:
        if (key == 'model_name') | (key == 'comment'):
            continue
        try:
            # If key doesn't exist, a KeyError will be thrown
            if params[key] == None:
                params[key] = old_params[key]
        except KeyError:
            # Key doesn't exist - populate with value from old params index
            params[key] = old_params[key]

    # Read in data
    if data == None:
        data = load_data(verbose=verbose)

    # Create generators for training, validation
    train_gen = WaveformGenerator(
            data.train, 
            batch_size=params['batch_size'], 
            balanced=True, 
            dp_prob=params['dp_prob'],
            decay=params['decay']
        )

    val_gen = WaveformGenerator(
            data.val, 
            batch_size=params['batch_size'], 
            balanced=True, 
            dp_prob=params['dp_prob'],
            decay=params['decay']
        )

    # Prepare callbacks
    callbacks = [train_gen, val_gen]

    if test == False:
        tb = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True)
        model_saver = ModelSaver(
                model, 'retrain', params, 
                comment="Retrained from {}".format(model_name),
                verbose=verbose, period=cp_interval
            )
        callbacks += [tb, model_saver]

    # Train model
    model.fit_generator(
            train_gen, 
            steps_per_epoch=params['steps_per_epoch'], 
            epochs=params['no_epochs'] - curr_epoch, 
            verbose=int(verbose), 
            validation_data=val_gen,
            validation_steps=params['steps_per_epoch'], 
            callbacks=callbacks,
            initial_epoch=curr_epoch
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
            Resumes training for a Keras model
            """
        )
    
    # Add arguments
    parser.add_argument(
            '-m', '--model-path', 
            help='path to the Keras HDF5 model file to be loaded',
            type=str, dest='filepath',
            required=True
        )

    parser.add_argument(
            '-c', '--curr-epoch', 
            help='the current number of epochs',
            type=int, dest='curr_epoch',
            required=True
        )

    parser.add_argument(
            '-e', '--no-epochs', 
            help='sets the total number of epochs (i.e. curr epoch + how many left to train for)',
            type=int, dest='no_epochs',
            required=True
        )

    parser.add_argument(
            '-l', '--learn-rate', 
            help='sets the learning rate for the Adam optimiser - if None, uses previous learning rate (default: None)',
            type=float, dest='lr', 
            default=None
        )

    parser.add_argument(
            '-b', '--batch-size', 
            help='sets the batch size',
            type=int, dest='batch_size', 
            default=None
        )

    parser.add_argument(
            '-s', '--steps-per-epoch', 
            help='sets the number of batches per epoch',
            type=int, dest='steps_per_epoch', 
            default=None
        )

    parser.add_argument(
            '-p', '--double-pulse-prob', 
            help='sets proportion of double pulse waveforms used at train time (default = 0.5)',
            type=float, dest='dp_prob', 
            default=None
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

    parser.add_argument(
            '-k', '--cp-interval', 
            help='sets number of epochs between the saving of model checkpoints',
            type=int, dest='cp_interval', 
            default=100
        )

    parser.add_argument(
            '-d', '--decay', 
            help='sets decay rate for the proportion of double-pulse waveforms used for training and validation (default = 0.0)',
            type=float, dest='decay', 
            default=None
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
        'batch_size': args.batch_size,
        'no_epochs': args.no_epochs,
        'steps_per_epoch': args.steps_per_epoch,
        'dp_prob': args.dp_prob,
        'decay': args.decay
    }

    main(
            args.filepath,
            args.curr_epoch,
            params=params,
            no_threads=args.no_threads,
            verbose=args.verbose,
            cp_interval=args.cp_interval,
            test=args.test
        )