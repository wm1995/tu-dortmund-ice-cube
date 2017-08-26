'''
This module defines a Keras generator class to provide batches for training

'''
from __future__ import division, print_function
from keras.utils import Sequence
from keras.callbacks import Callback

class WaveformGenerator(Sequence, Callback):
    '''
    Keras generator that provides batches for training and validation. If decay
    in proportions of double-pulse waveforms is used, the object must also be 
    passed as a callback when training a model. 

    Inherits from keras.utils.Sequence and keras.callbacks.Callback

    '''
    # keras.utils.Sequence object must implement __len__, __getitem__ methods
    
    FINAL_DP_PROB = 0.002 # The actual proportion of double pulse waveforms

    def __init__(
            self,
            data,
            batch_size,
            balanced=True,
            dp_prob=0.5,
            decay=0.0
        ):
        '''
        Constructor for WaveformGenerator class

        Arguments:
            data - dataset object
            batch_size - batch size to return
            balanced - if True, returns batches with double pulse waveforms
                       occurring with probability dp_prob; otherwise, returns 
                       batches without prior weighting between classes 
                       (default: True)
            dp_prob - the relative proportion (on average) of double pulse 
                      waveforms in batches with balanced = True (default: 0.5)
            decay - the amount of decay in dp_prob per epoch (default: 0.0)

        Returns:
            self - a WaveformGenerator object

        '''
        self.data = data
        self.batch_size = batch_size
        self.balanced = balanced
        self.dp_prob = dp_prob
        self.decay = decay

    def __len__(self):
        '''
        Returns number of batches in the current epoch

        No arguments

        Returns:
            n_batches - the number of batches in the current epoch

        '''
        return self.data.waveforms.shape[0] // self.batch_size

    def __getitem__(self, index):
        '''
        Returns next batch

        Arguments:
            index - does nothing but is required for inheritance reasons

        Returns: 
            waveforms - an array with shape [batch_size, 128] of waveform data
            labels - an array with shape [batch_size, 2], the corresponding 
                     labels for waveforms
        '''
        if self.balanced:
            return self.data.next_batch_balanced(
                    batch_size=self.batch_size,
                    dp_prob=self.dp_prob
                )
        else:
            return self.data.next_batch(batch_size=self.batch_size)
        
    def on_epoch_end(self, epoch, logs=None):
        '''
        Method called on the end of an epoch - decays db_prob after each epoch,
        i.e. gradually reduces the proportion of double pulse waveforms towards
        its true distribution.
        
        Arguments:
            epoch - a zero-indexed count of the current epoch
            logs - does nothing but is required for inheritance reasons
                   (default: None)

        No returns

        '''
        if self.decay != 0:
            self.dp_prob = (self.dp_prob - self.FINAL_DP_PROB) \
                               / (1 + self.decay * epoch) + self.FINAL_DP_PROB
