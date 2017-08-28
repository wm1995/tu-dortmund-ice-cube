'''
A module containing metrics for evaluating model performance after training - 
takes numpy arrays.

'''
from __future__ import print_function, absolute_import, division

from sklearn.metrics import confusion_matrix, roc_auc_score

from maxTools.sklearn_utils import \
    weighted_unique_confusion_matrix, weighted_confusion_matrix

# Most functions in this module are based off their unthresholded counterparts
# in maxTools.waveform_dataset

def threshold_confusion_matrix(y_true, y_pred, th=0.5):
    """
    Computes confusion matrix with a threshold in predictions.

    Takes numpy arrays
    
    Arguments:
        y_true - labels
        y_pred - predictions
        th - probability threshold above which the signal class is
             considered to predict signal (default: 0.5)
    
    Returns:
        confusion_matrix - a numpy array containing the confusion matrix

    """
    # This statement flattens vectors from one-hot, thresholds predictions
    return confusion_matrix(y_true.nonzero()[1], y_pred[:, 1] > th)

def threshold_weighted_confusion_matrix(y_true, y_pred, weights, th=0.5):
    """
    Computes a weighted confusion matrix with a threshold in predictions.

    Takes numpy arrays
    
    Arguments:
        y_true - labels
        y_pred - predictions
        weights - weights for each waveform
        th - probability threshold above which the signal class is
             considered to predict signal (default: 0.5)
    
    Returns:
        confusion_matrix - a numpy array containing the confusion matrix

    """
    # This statement flattens vectors from one-hot, thresholds predictions
    return weighted_confusion_matrix(y_true.nonzero()[1],
                                         y_pred[:, 1] > th, weights)

def threshold_weighted_unique_confusion_matrix(y_true, y_pred,
                                               weights, ids, th=0.5):
    """
    Computes a weighted event-wise confusion matrix with a threshold in 
    predictions.

    Takes numpy arrays
    
    Arguments:
        y_true - labels
        y_pred - predictions
        weights - weights for each waveform
        ids - ids to correlate waveforms with events
        th - probability threshold above which the signal class is
             considered to predict signal (default: 0.5)
    
    Returns:
        confusion_matrix - a numpy array containing the confusion matrix

    """
    # This statement flattens vectors from one-hot, thresholds predictions
    return weighted_unique_confusion_matrix(y_true.nonzero()[1],
                                            y_pred[:, 1] > th, weights, ids)

def metric_summary(y_true, y_pred, th=0.5):
    """
    Computes a summary of different metrics at a given threshold. The metrics
    computed are accuracy, precision, recall and F1 score.

    Takes numpy arrays
    
    Arguments:
        y_true - labels
        y_pred - predictions
        th - probability threshold above which the signal class is
             considered to predict signal (default: 0.5)
    
    Returns:
        accuracy - a float
        precision - a float
        recall - a float
        f1 - a float

    """
    [[tn, fp], [fn, tp]] = threshold_confusion_matrix(y_true, y_pred, th)

    accuracy = (tn + tp) / (tn + tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1

def print_metric_results(y_true, y_pred, weights, ids, th=0.5):
    """
    Prints a summary output of different metrics.

    Takes numpy arrays
    
    Arguments:
        y_true - labels
        y_pred - predictions
        weights - weights for each waveform
        ids - ids to correlate waveforms with events
        th - probability threshold above which the signal class is
             considered to predict signal (default: 0.5)
    
    No returns
    
    """
    accuracy, precision, recall, f1 = metric_summary(y_true, y_pred, th)

    print("For a threshold of {}:".format(th))
    
    print()

    print("\tAccuracy:       ", accuracy)
    print("\tPrecision:      ", precision)
    print("\tRecall:         ", recall)
    print("\tF1 Score:       ", f1)
    print("\tAUC             ", roc_auc_score(y_true, y_pred))

    print()

    print("\tRaw confusion matrix:")
    
    [[tn, fp], [fn, tp]] = threshold_confusion_matrix(y_true, y_pred, th)
    
    print("\t\t%8d\t%8d" % (tn, fp))
    print("\t\t%8d\t%8d" % (fn, tp))

    print()

    print("\tWeighted confusion matrix:")
    
    [[tn, fp], [fn, tp]] = threshold_weighted_confusion_matrix(
        y_true, y_pred, weights, th)
    
    print("\t\t%.3e\t%.3e" % (tn, fp))
    print("\t\t%.3e\t%.3e" % (fn, tp))

    print()

    print("\tWeighted unique confusion matrix:")
    
    [[tn, fp], [fn, tp]] = threshold_weighted_unique_confusion_matrix(
        y_true, y_pred, weights, ids, th)
    
    print("\t\t%.3e\t%.3e" % (tn, fp))
    print("\t\t%.3e\t%.3e" % (fn, tp))

    print()
