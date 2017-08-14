#!/usr/bin/env python
# Contains definition for Keras generator using balanced generator

from keras.utils import Sequence

class WaveformGenerator(Sequence):
    '''Object to give the next batch to Keras for training purposes

    Inherits from keras.utils.Sequence

    Must implement the __len__ and __getitem__ methods

    '''
    def __init__(self, data, batch_size, balanced=True, dp_prob=0.5):
        '''Constructor

        Arguments:
            data - dataset object
            batch_size - batch size to return
            balanced - if True, returns balance of waveforms as dictated by dpProb
            dp_prob - the relative proportion (on average) of double pulse waveforms

        '''
        self.data = data
        self.batch_size = batch_size
        self.balanced = balanced
        self.dp_prob = dp_prob

    def __len__(self):
        '''Number of batches in the sequence

        May need reimplementing...
        
        '''
        return self.data.waveforms.shape[0] // self.batch_size

    def __getitem__(self, index):
        '''Returns next batch

        Arguments:
            index - does nothing but is required

        '''
        if self.balanced:
            return self.data.next_batch_balanced(batch_size=self.batch_size, dp_prob=self.dp_prob)
        else:
            return self.data.next_batch(batch_size=self.batch_size)
