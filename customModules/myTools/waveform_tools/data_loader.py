'''
A module to load data and scale appropriately (based on methods and classes in 
maxTools.waveform_dataset)

'''
import cPickle as pickle
import numpy as np

from maxTools import waveform_dataset

from sklearn.preprocessing import StandardScaler

SCALE_FILEPATH = "scale.p"

def save_scaling(data, filepath):
    '''
    Calculates the scaling necessary to rescale data to have a mean of 0 and a
    variance of 1, then pickles this scaling to the file given by filepath.

    Arguments:
        data - an array of shape [n, 128] of data for which the scaling is to
               be calculated
        filepath - the path where the pickled scaling is to be saved

    '''
    scaler = StandardScaler()
    scaler.fit(data)
    pickle.dump(scaler.scale_, open(filepath, 'w'))

def scale_data(data, scale=None):
    '''
    Scales waveform data to have a mean of 0 and a variance of 1, using scaling
    calculated from the training data waveforms. The data is scaled in place, 
    i.e. no copy is made or returned. 

    Arguments:
        data - an array of shape [n, 128] of data to be scaled
        scale - an array of scalings of shape [128], if None then a default
                scaling is loaded from a pickled object (default: None)

    No returns

    '''
    if scale is None:
        scale = pickle.load(open(SCALE_FILEPATH))

    np.multiply(data, scale, out=data)

def load_data(verbose=True, train_ratio=0.8, test_ratio=0.13, rescale=True):
    '''
    Load data using methods in maxTools.waveform_dataset, preprocess it, and
    return the resulting Datasets object

    The data is preprocessed to have mean of 0 and variance of 1. 

    Arguments:
        verbose - print data about scaling
        train_ratio - ratio of data loaded designated for training
        test_ratio - ratio of data loaded set aside for testing

    Returns: 
        data - a Datasets object, a named tuple of three DataSet objects 
               (train, val and test), each with the following structure:
            waveforms - an array of shape [n, 128] with the waveform data
            labels - a one-hot array of shape [n, 2] with the correct labels
            weights - an array of shape [n] with weights for events
            ids - an array of shape [n] with ids for events
            methods for generating batches of data
            
    For further information on the DataSet object, refer to its definition in
    maxTools.waveform_dataset

    '''
    # Selected datasets
    datasets = [
        ('11538', ['DP', 'NC'], 3638),  # Tau
        ('12034', ['CC', 'NC'], 8637),  # Electron
        ('11069', ['NC'], 7287)         # Muon
        # NB - discard muon track events because of similarity to dp events
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

    # Rescale input data to give training data mean 0 and stdev 1
    if rescale is True:
        for dataset in data:
            scale_data(dataset.waveforms)
        
    return data

def load_eval_data(verbose=True, train_ratio=0.8, test_ratio=0.13, rescale=True):
    ''' 
    Load evaluation data using methods in maxTools.waveform_dataset, preprocess
    it, and return the resulting Datasets object

    The data is preprocessed to have mean of 0 and variance of 1. 

    Arguments:
        verbose - print data about scaling
        train_ratio - ratio of data loaded designated for training
        test_ratio - ratio of data loaded set aside for testing

    Returns: 
        data - a Datasets object, a named tuple of three DataSet objects 
               (train, val and test), each with the following structure:
            waveforms - an array of shape [n, 128] with the waveform data
            labels - a one-hot array of shape [n, 2] with the correct labels
            weights - an array of shape [n] with weights for events
            ids - an array of shape [n] with ids for events
            methods for generating batches of data
            
    For further information on the DataSet object, refer to its definition in
    maxTools.waveform_dataset

    '''
    # Selected datasets
    datasets = [
        ('11538', ['DP', 'NDP', 'NC'], 3638),  # Tau neutrinos
        ('12034', ['CC', 'NC'], 8637),  # Electron neutrinos
        ('11069', ['NC', 'CC'], 7287),   # Muon neutrinos
        ('11057', ['AM'], 74890)        # Atmospheric muons
    ]

    # Read in data - output is split into train, val and test
    if verbose:
        print("Loading Data...")
    data = waveform_dataset.read_data(
            datasets, 
            combined=True, 
            train_ratio=train_ratio, 
            test_ratio=test_ratio,
            load_eval_data=True,
            verbose=verbose
        )

    # Rescale input data to give training data mean 0 and stdev 1
    if rescale is True:
        for dataset in data:
            scale_data(dataset.waveforms)
        
    return data
