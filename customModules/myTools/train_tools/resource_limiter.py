'''
Module containing methods to limit the amount of resources Keras and 
Tensorflow will use

'''
import tensorflow as tf
from keras import backend as K

def limit_resources(no_threads=10, limit_gpu=True):
    '''
    Sets Keras to use a session that limits the resources used

    Arguments:
        no_threads - limits maximum number of threads, if 0 then no limit
                     NB max no of threads > no_threads (generally x2)

    No returns

    '''
    # Set up CPU, GPU options
    config = tf.ConfigProto(
            intra_op_parallelism_threads=no_threads, 
            inter_op_parallelism_threads=no_threads,
            allow_soft_placement=True,
            gpu_options = tf.GPUOptions(allow_growth = limit_gpu)
        )

    sess = tf.Session(config=config)
    K.set_session(sess)
