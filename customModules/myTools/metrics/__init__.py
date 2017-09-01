''' 
A package containing metrics for evaluating the models both during and after
training.

Contains:
    keras - a module containing metrics for use with Keras during training,
            takes tf.Tensor objects
    sklearn - a module containing metrics for evaluating the model after 
              training, takes numpy arrays

'''

__all__ = ['keras', 'sklearn']
