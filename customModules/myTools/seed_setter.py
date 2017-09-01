'''
Module for setting all the necessary seeds for Keras

'''
import os
import numpy as np
import random as rn
import tensorflow as tf

def set_seed(s):
    '''
    Sets all random seeds used by Keras

    Arguments:
        s - an integer, the seed

    No returns

    '''
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    rn.seed(s)
    tf.set_random_seed(s)

def check_seed_set():
    '''
    Checks if environment variable RANDOM_SEED is set, and if so, sets the 
    random seeds. 

    No arguments

    No returns
    
    '''
    # If environment variable RANDOM_SEED is set, set seed
    RANDOM_SEED = os.environ.get('RANDOM_SEED')

    if RANDOM_SEED != None:
        set_seed(int(RANDOM_SEED))
