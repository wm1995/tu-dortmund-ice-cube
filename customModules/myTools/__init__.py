'''
A package containing the tools I have written to work with the waveform data 
and Keras

Contains:
    metrics - package of metrics for Keras and post-training evaluation
    model_tools - package of tools for saving and loading models
    waveform_tools - package of tools for loading the data and feeding batches
                     to Keras
    seed_setter - module for setting Keras random seed

'''

__all__ = ['metrics', 'model_tools', 'waveform_tools', 'seed_setter']
