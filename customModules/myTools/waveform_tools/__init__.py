'''
A package containing tools necessary for loading the data and generating 
batches for Keras for training. 

Contains: 
    data_loader - a module containing a method for loading the waveform
                  data into memory
    waveform_generator - a module containing a keras.utils.Generator 
                         object for supplying batches during training
                         and validation

'''

__all__ = ["data_loader", "waveform_generator"]
