'''
Module containing a function that sets all the necessary seeds for Keras

'''
import os
import numpy as np
import random as rn
import tensorflow as tf

def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    rn.seed(s)
    tf.set_random_seed(s)

def check_seed_set():
    # If environment variable RANDOM_SEED is set, set seed
    RANDOM_SEED = os.environ.get('RANDOM_SEED')

    if RANDOM_SEED != None:
        set_seed(int(RANDOM_SEED))
