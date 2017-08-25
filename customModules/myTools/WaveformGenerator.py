#!/usr/bin/env python
# Contains definition for Keras generator using balanced generator
from __future__ import division, print_function
from keras.utils import Sequence
from keras.callbacks import Callback

class WaveformGenerator(Sequence, Callback):
    '''Object to give the next batch to Keras for training purposes

    Inherits from keras.utils.Sequence, keras.callbacks.Callback

    Must implement the __len__ and __getitem__ methods

    '''
    FINAL_DP_PROB = 0.002 # The actual proportion of double pulse waveforms

    def __init__(self, data, batch_size, balanced=True, dp_prob=0.5, decay=0.0):
        '''Constructor

        Arguments:
            data - dataset object
            batch_size - batch size to return
            balanced - if True, returns balance of waveforms as dictated by dpProb
            dp_prob - the relative proportion (on average) of double pulse waveforms
            decay - the amount of decay in dp_prob per epoch

        '''
        self.data = data
        self.batch_size = batch_size
        self.balanced = balanced
        self.dp_prob = dp_prob
        self.decay = decay

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
        
    def on_epoch_end(self, epoch, logs=None):
        if self.decay != 0:
            self.dp_prob = (self.dp_prob - self.FINAL_DP_PROB) / (1 + self.decay * epoch) + self.FINAL_DP_PROB