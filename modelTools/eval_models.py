from __future__ import division, print_function

import numpy as np
import argparse

from myTools.waveform_tools.data_loader import load_data, load_eval_data
from myTools.model_tools.model_loader import load_model
from myTools.metrics.sklearn import print_metric_results
from myTools.metrics.plots import purity_efficiency_plot, rate_plot

import tensorflow as tf
from keras import backend as K

def main(
        model_list,
        data=None,
        no_threads=10,
        verbose=True,
        val=True
    ):
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

    # Read in data
    if data == None:
        data = load_eval_data(verbose=verbose)

    for model_path in model_list:
        if verbose:
            print('Loading ' + model_path)
	# Load model
        model = load_model(model_path)

        # Get the name of the model
        model_name = model_path.split('/')[-1]
        model_name = model_name[:-3]          # Strip off extension
    
        secs_per_year = 86400 * 365.25

        # Evaluate model
        if val:
            dataset = data.val
            data_ratio = 0.07
        else:
            dataset = data.test
            data_ratio = 0.13

        # Rescale weights
        weights = dataset.weights * secs_per_year / data_ratio
            
        test_preds = model.predict(dataset.waveforms, verbose=int(verbose))

        training_mask = np.logical_or(dataset.labels[:, 0] == 1, dataset.labels[:, 1] == 1)

        print()
        print_metric_results(dataset.labels[training_mask, 0:2], test_preds[training_mask, 0:2], weights[training_mask], dataset.ids[training_mask], th=0.5)
        print_metric_results(dataset.labels[training_mask, 0:2], test_preds[training_mask, 0:2], weights[training_mask], dataset.ids[training_mask], th=0.9)
	print()

        purity_efficiency_plot(dataset.labels[training_mask, 0:2], test_preds[training_mask, 0:2], savepath="plots/" + model_name + "_pe_plot.pdf")
        rate_plot(dataset.labels[training_mask, 0:2], test_preds[training_mask, 0:2], weights[training_mask], savepath="plots/" + model_name + "_train_rate_plot.pdf")
        rate_plot(dataset.labels, test_preds, weights, combine_nu_tau_cc=True, savepath="plots/" + model_name + "_rate_plot.pdf")

if __name__ == "__main__":
    # Initialise the arg parser
    parser = argparse.ArgumentParser(
            description="""
            Evaluates model(s) with validation or test data
            """
        )
    
    # Add mutually exclusive val/test data group
    arg_group = parser.add_mutually_exclusive_group()

    arg_group.add_argument(
            '--val', 
            help='uses validation data for evaluation',
            action='store_true', dest='val', 
            default=False
        )

    arg_group.add_argument(
            '--test', 
            help='uses test data for evaluation',
            action='store_true', dest='test', 
            default=False
        )

    # Set number of threads
    parser.add_argument(
            '-t', '--no-threads', 
            help='sets limit on the number of threads to be used (default = 10, if no limit set to -1)',
            type=int, dest='no_threads', 
            default=10
        )

    # Get models
    parser.add_argument(
            'model_filepaths', nargs='+', 
            help='paths to the Keras HDF5 model files to be evaluated',
            type=str
        )

    # Parse the args
    args = parser.parse_args()

    main(
            args.model_filepaths,
            no_threads=args.no_threads,
            val=args.val
        )
