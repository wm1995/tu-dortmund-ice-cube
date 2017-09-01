''' 
A package containing tools for saving models and a short description of their 
parameters to the model store, and tools for reloading the models from the
model store. 

Contains: 
    model_saver - a module containing a Keras callback object that saves models
                  during and after training
    model_loader - a module that loads previously saved models

'''

__all__ = ['model_saver', 'model_loader']
