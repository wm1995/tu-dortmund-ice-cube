#!/usr/bin/env python
# Module to save Keras models to the model store

from __future__ import print_function

import csv
import time

# Constants
MODEL_DIR = '/fhgfs/users/wmartin/models/'    # Path to model directory
MODEL_SUMMARY = 'model_summary.csv'           # CSV summary of models

def save_model(model, nn_str, params, comment="", verbose=False):
    """
    Saves a Keras model to the directory of models, updates the CSV summary with the model parameters

    Arguments
        model - the Keras model to be saved
        nn_str - the string specifying what type of model was trained (e.g. "cnn", "snn", etc.)
                 (only used for the filename)
        params - the parameter dictionary for the model
                 (for format see the Keras cnn.py or snn.py)
        comment - string to put in comments field in CSV (default: "")
        verbose - if True, prints when model is saved (default: False)

    No returns
    """
    # Save model
    datetime = time.strftime("%Y%m%d_%H%M%S_")
    save_path = MODEL_DIR + datetime + nn_str + 'Keras.h5'
    model.save(save_path)
    
    if verbose:
        print("Model saved to " + save_path)

    # Update CSV file
    with open(MODEL_DIR + MODEL_SUMMARY, 'a+') as csvfile:
        # Read file to extract headers
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames

        # Add comment to parameter dictionary
        params['comments'] = comment

        # Write param dictionary to file
        writer = csv.DictWriter(csvfile, headers)
        writer.writerow(params)

    if verbose:
        print("Model summary updated")
