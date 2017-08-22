#!/usr/bin/env python
# Module to save Keras models to the model store

from __future__ import print_function

import csv
import time

from keras.callbacks import ModelCheckpoint

# Constants
MODEL_DIR = '/fhgfs/users/wmartin/models/'    # Path to model directory
MODEL_SUMMARY = 'model_summary.csv'           # CSV summary of models

class ModelSaver(ModelCheckpoint):
    """
    A Keras callback that saves the model to the model store at regular intervals.

    The function saves the model when instantiated, at specified intervals during training,
    and at the end of training. 

    Inherits from keras.callbacks.ModelCheckpoint

    Arguments
        model - the Keras model to be saved
        nn_str - the string specifying what type of model was trained (e.g. "cnn", "snn", etc.)
                 (only used for the filename - no commas or forward-slashes!)
        params - the parameter dictionary for the model, of the following format
            lr - the learning rate of the Adam optimiser
            conv_dr - the dropout rate after the convolutional layers
            fc_dr - the dropout rate after the fully-connected layers
            no_epochs - the number of epochs to run for
            steps_per_epoch - the number of batches in each epoch
            dp_prob - the proportion of double pulse waveforms shown at train time
            batch_norm - if true, use batch norm after each layer
            regularise - if true, uses L2 regularisation on the weights for each layer
        comment - string to put in comments field in CSV (default: "")
                  (commas will be replaced with semicolons!)
        period - number of epochs between checkpoint saves (default: 100)
        verbose - if True, prints when model is saved (default: False)
    """

    def _save_model(self):
        self.model.save(self.filepath, overwrite=True)
        if self.verbose > 0:
            print("Model saved to " + self.filepath)

    def _write_summary(model_name, params, comment):
        # Update CSV file
        with open(MODEL_DIR + MODEL_SUMMARY, 'a+') as csvfile:
            # Read file to extract headers
            reader = csv.DictReader(csvfile)
            fields = reader.fieldnames

            # Add model_name to parameter dictionary
            # Replace all commas, forward slashes with underscores
            params['model_name'] = model_name.replace(",", "_").replace("/", "_")
            
            # Add comment to parameter dictionary
            # Replace all commas with semicolons
            params['comments'] = comment.replace(",", ";")

            # Write param dictionary to file
            writer = csv.DictWriter(csvfile, fields)
            writer.writerow(params)

        if self.verbose > 0:
            print("Model added to summary")

    def __init__(self, model, nn_str, params, comment="", period=100, verbose=False):
        # Create savepath
        datetime = time.strftime("%Y%m%d_%H%M%S_")
        model_name = datetime + nn_str + 'Keras.h5'
        filepath = MODEL_DIR + model_name

        # Initialise the parent class ModelSaver
        super(ModelSaver, self).__init__(
                filepath=filepath, 
                verbose=int(verbose), 
                period=period
            )

        # Set, save model
        self.set_model(model)
        self._save_model()

        # Update CSV file listing models
        self._write_summary(model_name, params, comment)

    def on_train_end(self, logs=None):
        self._save_model()


    

