#!/usr/bin/env python
# Module to load data and scale appropriately (based on Max's dataloader)
import numpy as np

from maxTools import waveform_dataset

from sklearn.preprocessing import StandardScaler

def load_data(verbose=True, train_ratio=0.8, test_ratio=0.13, rescale=True):
    ''' 
    Load data using Max's script, normalise, and return data object

    The data is normalised in the preprocessing stage to have mean of 0 and variance of 1. 

    Arguments:
        verbose - print data about scaling
        train_ratio - ratio of data loaded designated for training
        test_ratio - ratio of data loaded set aside for testing

    Returns: 
    The final object returned is a Datasets object, with the following structure:
        - The object has three Dataset objects (train, val and test)
        - Each of the Dataset objects has:
            - waveforms - an array of shape [n, 128] with the waveform data
            - labels - a one-hot array of shape [n, 2] with the correct labels
                - Note that there is a large imbalance in the data
                      99.8% noise vs 0.2% signal
            - weights - an array of shape [n] with weights for events
            - ids - an array of shape [n] with ids for events

    '''
    # Selected datasets
    datasets = [
        ('11538', ['DP', 'NC'], 3638),  # Tau
        ('12034', ['CC', 'NC'], 8637),  # Electron
        ('11069', ['NC'], 7287)         # Muon (discard muon track events because of similarity to dp events)
    ]

    # Read in data - output is split into train, val and test
    if verbose:
        print("Loading Data...")
    data = waveform_dataset.read_data(
            datasets, 
            combined=True, 
            train_ratio=train_ratio, 
            test_ratio=test_ratio,
            verbose=verbose
        )

    # if verbose:
    #     print("Before scaling:        \tMean       \tMean Stdev")

    #     print("    Training data:    \t%10.3f \t%10.3f" % (
    #             np.mean(data.train.waveforms), 
    #             np.mean(np.std(data.train.waveforms, axis=0))
    #         ))

    #     print("  Validation data:    \t%10.3f \t%10.3f" % (
    #             np.mean(data.val.waveforms), 
    #             np.mean(np.std(data.val.waveforms, axis=0))
    #         ))

    #     print("        Test data:    \t%10.3f \t%10.3f" % (
    #             np.mean(data.test.waveforms), 
    #             np.mean(np.std(data.test.waveforms, axis=0))
    #         ))

    # Rescale input data to give training data mean 0 and stdev 1
    if rescale is True:
        rescaler = StandardScaler(copy=False)
        rescaler.fit_transform(data.train.waveforms)
        rescaler.transform(data.val.waveforms)
        rescaler.transform(data.test.waveforms)

    # if verbose:
    #     print("Data rescaled:        \tMean       \tMean Stdev")

    #     print("    Training data:    \t%10.3f \t%10.3f" % (
    #             np.mean(data.train.waveforms), 
    #             np.mean(np.std(data.train.waveforms, axis=0))
    #         ))

    #     print("  Validation data:    \t%10.3f \t%10.3f" % (
    #             np.mean(data.val.waveforms), 
    #             np.mean(np.std(data.val.waveforms, axis=0))
    #         ))

    #     print("        Test data:    \t%10.3f \t%10.3f" % (
    #             np.mean(data.test.waveforms), 
    #             np.mean(np.std(data.test.waveforms, axis=0))
    #         ))
        
    return data
