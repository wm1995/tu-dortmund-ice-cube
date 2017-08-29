from keras.models import load_model as _load_model
from myTools.metrics.keras import precision, recall, f1, class_balance

# Constants
# Need to specify the custom objects used in model compilation
CUSTOM_OBJ_DICT = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_balance': class_balance
    }

def load_model(filepath):
    '''
    Loads and compiles the specified Keras model.

    NB Model must have been pre-compiled with an optimiser

    Arguments:
        filepath - the filepath of the model

    Returns:
        model - a keras.models.Sequential object

    '''
    return _load_model(filepath, custom_objects=CUSTOM_OBJ_DICT)

def load_uncompiled_model(filepath):
    '''
    Loads the specified Keras model without compiling it.

    NB Model can be either compiled or uncompiled.

    Arguments:
        filepath - the filepath of the model

    Returns:
        model - a keras.models.Sequential object

    '''
    return _load_model(filepath, compile=False)


