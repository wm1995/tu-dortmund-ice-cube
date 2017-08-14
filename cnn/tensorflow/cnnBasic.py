#!/usr/bin/env python
# Convolutional neural network to distinguish double-pulse events
# William Martin
# 26/07/2017

# Code structure based on https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
# (Accessed 26/07/2017) Also the source for the flags code

from __future__ import division, print_function
import numpy as np

import tensorflow as tf

from maxTools import waveform_dataset

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# # Following flags code is borrowed from the cifar10 example from TensorFlow (see above)
# FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_boolean('use_fp16', False,
#                             """Use 16-bit floating pt numbers.""")

def loadData(verbose = False, train_ratio = 0.8, test_ratio = 0.13):
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
    data = waveform_dataset.read_data(datasets, combined=True, train_ratio = train_ratio, test_ratio = test_ratio)

    if verbose:
        print("Before scaling:        \tMean       \tMean Stdev")
        print("    Training data:    \t%10.3f \t%10.3f" % (np.mean(data.train.waveforms), np.mean(np.std(data.train.waveforms, axis = 0))))
        print("  Validation data:    \t%10.3f \t%10.3f" % (np.mean(data.val.waveforms), np.mean(np.std(data.val.waveforms, axis = 0))))
        print("        Test data:    \t%10.3f \t%10.3f" % (np.mean(data.test.waveforms), np.mean(np.std(data.test.waveforms, axis = 0))))

    # Rescale input data to give training data mean 0 and stdev 1
    rescaler = StandardScaler(copy = False)
    rescaler.fit_transform(data.train.waveforms)
    rescaler.transform(data.val.waveforms)
    rescaler.transform(data.test.waveforms)

    if verbose:
        print("Data rescaled:        \tMean       \tMean Stdev")
        print("    Training data:    \t%10.3f \t%10.3f" % (np.mean(data.train.waveforms), np.mean(np.std(data.train.waveforms, axis = 0))))
        print("  Validation data:    \t%10.3f \t%10.3f" % (np.mean(data.val.waveforms), np.mean(np.std(data.val.waveforms, axis = 0))))
        print("        Test data:    \t%10.3f \t%10.3f" % (np.mean(data.test.waveforms), np.mean(np.std(data.test.waveforms, axis = 0))))

    return data

def _generateWeights(shape, n_in):
    ''' 
    Returns weights that are set up for ReLU neurons
    Allows for L2 Regularisation

    Weights are normally distributed with variance 2 / n_in
    (Modified Xavier initialisation - He et al. 2015, arXiv:1502.01852)
    
    Arguments:
        shape - shape of weights tensor
        n_in - the number of connected inputs (i.e. the volume of the previous layer)

    Returns: 
        TensorFlow variable (with shape as specified by argument)

    ''' 
    stdev = np.sqrt(2 / n_in)
    weights = tf.Variable(tf.random_normal(shape, stddev = stdev, dtype = tf.float32), name = 'weights')

    # Regularisation loss code taken from CIFAR-10 tf example
    regLoss = tf.nn.l2_loss(weights, name = 'weights_loss')
    tf.add_to_collection('losses', regLoss)
    return weights

def _generateBiases(noLayers, b_0 = 0):
    '''
    Returns biases initialised to b_0
    
    Set b_0 positive and small for ReLu neurons
    
    Arguments:
        noLayers - number of layers required (i.e. the dimension of the bias)
        b_0 - initial value to initialise the bias with

    Returns:
        TensorFlow variable (column vector with noLayers dimensions)

    '''
    return tf.Variable(tf.constant(b_0, shape = [noLayers], dtype = tf.float32), name = 'biases')

def _conv(name, inputTensor, n_in, noLayersIn, noLayersOut, filtSize = 3, stride = 1, padding = 'SAME'):
    ''' 
    Returns a convolutional layer with ReLU activation.

    Arguments:
        name - name of layer
        inputTensor - the tensor that the layer operates on
        n_in - number of connected inputs (i.e. volume of previous layer)
        noLayersIn - depth of input
        noLayersOut - depth of output
        filtSize - size of filter
        stride - stride
        padding - type of padding - either 'SAME' (maintain size) or 'VALID' (no padding)

    Returns:
        tf.Tensor object - result of the conv layer computation; shape = (-1, n, noLayersOut)

    '''
    with tf.variable_scope(name) as scope:  
        weights = _generateWeights([filtSize, noLayersIn, noLayersOut], n_in)
        biases = _generateBiases(noLayersOut, b_0 = 0)
        conv = tf.nn.conv1d(inputTensor, weights, stride, padding)
        pre_activation = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(pre_activation, name = scope.name)
    return activation

def _dense(name, inputTensor, depthIn, depthOut, activation = tf.nn.relu):
    ''' 
    Returns a fully-connected layer with the specified activation
    
    Arguments: 
        name - name of layer
        inputTensor - the tensor that the layer operates on
        depthIn - depth of input
        depthOut - depth of output
        activation - activation (tf object)

    Returns:
        tf.Tensor object - result of dense layer with specified activation; shape = (-1, depthOut)

    '''
    with tf.variable_scope(name) as scope:
        weights = _generateWeights([depthIn, depthOut], depthIn)
        biases = _generateBiases(depthOut, b_0 = 0)
        dense = tf.matmul(inputTensor, weights)
        output = None
        if activation == None:
            output = tf.nn.bias_add(dense, biases, name = scope.name)
        else:
            pre_activation = tf.nn.bias_add(dense, biases)
            output = activation(pre_activation, name = scope.name)
        return output


def loss(logits, labels):
    ''' 
    Gives loss for training step - uses L2 regularisation on weights
    Also evaluates precision, attaches summary
    
    Arguments:
        logits - input array of raw logits from inference()
        labels - labels of batch

    Returns:
        loss - tf.float32, sum of cross entropy and L2 regularisation losses

    '''
    # Code taken from CIFAR-10 tf example
    # Calculate cross-entropy loss
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # Add up all losses (i.e. cross-entropy and L2 loss)
    loss = tf.add_n(tf.get_collection('losses'))
    return loss

def main(data = None, lr = 1e-4, convkp = 0.5, fckp = 0.5, b_0 = 0):
    # Load data
    # (Temporary rewrite for debugging convenience)
    if data == None:
        data = loadData()
    
    # Construct graph
    # Set up input variables
    # x = input waveform, yPred = output prediction
    x = tf.placeholder(tf.float32, shape = [None, 128])
    yTrue = tf.placeholder(tf.float32, shape = [None, 2])

    xCorr = tf.reshape(x, [-1, 128, 1])
 
    # Convolutional layer
    conv1 = _conv('conv1', xCorr, n_in = 128 * 1, noLayersIn = 1, noLayersOut = 64, filtSize = 5)

    # Dropout layer
    drop1 = tf.nn.dropout(conv1, keep_prob = convkp, name = "drop1")

    # Convolutional layer    
    conv2 = _conv('conv2', drop1, n_in = 128 * 64, noLayersIn = 64, noLayersOut = 128, filtSize = 3)
    
    # Dropout layer
    drop2 = tf.nn.dropout(conv2, keep_prob = convkp, name = "drop2")

    # Convolutional layer
    conv3 = _conv('conv3', drop2, n_in = 128 * 128, noLayersIn = 128, noLayersOut = 64, filtSize = 3)

    # Dropout layer
    drop3 = tf.nn.dropout(conv3, keep_prob = convkp, name = "drop3")

    # Flatten for fully connected layers
    drop3Flat = tf.reshape(drop3, [-1, 128 * 64])

    # Fully-connected layer
    fc4 = _dense('fc4', drop3Flat, depthIn = 128 * 64, depthOut = 1024)

    # Dropout layer
    drop4 = tf.nn.dropout(fc4, keep_prob = fckp, name = "drop4")

    # Fully-connected layer
    fc5 = _dense('fc5', drop4, depthIn = 1024, depthOut = 1024)
    
    # Dropout layer
    drop5 = tf.nn.dropout(fc5, keep_prob = fckp, name = "drop5")
    
    # Fully-connected layer
    fc6 = _dense('fc6', drop5, depthIn = 1024, depthOut = 1024)
    
    # Dropout layer
    drop6 = tf.nn.dropout(fc6, keep_prob = fckp, name = "drop6")
    
    ## fc6 = tf.layers.dense(inputs = fc5, units = 1024, activation = tf.nn.relu, name = "fc6")

    # Linear classifcation layer - apply softmax later for efficiency
    # Using tf CIFAR-10 structure for softmax
    logits = _dense('logits', drop6, depthIn = 1024, depthOut = 2, activation = None)

    yPred = tf.nn.softmax(logits)
    currLoss = loss(logits, yTrue)

    # Set up training step
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(currLoss)
    
    # Calculate precision and attach a scalar to it
    # precision = tf.metrics.precision(yTrue, yPred)
    precision = tf.metrics.precision_at_thresholds(yTrue, yPred, (0.5, 0.5))

    # Set up CPU, GPU options
    config = tf.ConfigProto(
            intra_op_parallelism_threads=10, 
            inter_op_parallelism_threads=10,
            allow_soft_placement=True, 
            device_count = {'CPU': 1, 'GPU': 1}, 
            gpu_options = tf.GPUOptions(allow_growth = True)
        )

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # All heavily adapted from tf TensorBoard tutorial
        for i in range(200):
            # Get the batch
            batchWaveforms, batchLabels = data.train.next_batch_balanced(batch_size = 64)
            
            # Run training step
            sess.run(train_step, 
                        feed_dict = {
                            x: batchWaveforms,
                            yTrue: batchLabels
                        })
            # valLog.add_summary(summary, i)
            # summary, _ = sess.run([tbSummary, train_step], 
            #             feed_dict = {
            #                 x: batchWaveforms,
            #                 yTrue: batchLabels
            #             })
            # valLog.add_summary(summary, i)
            # if i % 10 == 0:
            #     # Every 10th step, run validation
            #     summary, precision = sess.run([tbSummary, precision], 
            #                 feed_dict = {
            #                     x: data.val.waveforms,
            #                     yTrue: data.val.labels
            #                 })
            #     valLog.add_summary(summary, i)
            #     print('Precision at step %s: %s' % (i, precision))


    # What do we need to do?
    # Need training loss, training precision, validation precision (Need to output these things too!)
    # Need to loop over the training step several thousand times (maybe just 100 when debugging)

# Thoughts for how to implement moving forward?
# Have basic model layout, need training, loss
#
# Training
#   adam descent optimiser, learning rate
#
# Loss
# accuracy is an entirely useless parameter for evaluation in such an unbalanced dataset
# Do validation on unbalanced dataset or balanced dataset?
# 

# Pre-weight training batches, loss is then OK
# In preweighting, how to average no of batches sampled? use binomial distribution 


if __name__ == "__main__":
    #Temp rewrite for debugging convenience
    data = loadData()
    main(data = data)