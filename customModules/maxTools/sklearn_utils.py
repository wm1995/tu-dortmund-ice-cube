import numpy as np


def weighted_confusion_matrix(y_true, y_pred, weights):
    '''
    This is a modification of sklearn.metrics.confusion_matrix
    to incorporate weights.
    '''
    weights = np.asarray(weights)
    assert y_true.shape[0] == weights.shape[0], (
        'Labels and weights should have the same shape')
    from sklearn.utils.multiclass import unique_labels
    labels = unique_labels(y_true, y_pred)
    n_labels = labels.size

    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]

    from scipy.sparse import coo_matrix
    CM = coo_matrix((weights, (y_true, y_pred)),
                    shape=(n_labels, n_labels)
                    ).toarray()

    return CM


def weighted_unique_confusion_matrix(y_true, y_pred, weights, ids):
    '''
    This is a modification of sklearn.metrics.confusion_matrix
    to incorporate weights. This return an eventwise (not waveformwise)
    confusion matrix.
    '''
    weights = np.asarray(weights)
    assert y_true.shape[0] == weights.shape[0], (
        'Labels and weights should have the same shape')
    from sklearn.utils.multiclass import unique_labels
    labels = unique_labels(y_true, y_pred)
    n_labels = labels.size

    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]

    assert n_labels == 2, (
        'This shouldnt be used with more than 2 labels, '
        'adjust this fnctn first')
    sorted_ind = np.argsort(ids)
    u_val, u_ind = np.unique(ids[sorted_ind], return_index=True)
    y_true_evt = y_true[sorted_ind][u_ind]
    y_pred_evt = np.add.reduceat(y_pred[sorted_ind], u_ind)
    y_pred_evt = np.clip(y_pred_evt, 0, 1)
    weights = weights[sorted_ind][u_ind]

    from scipy.sparse import coo_matrix
    CM = coo_matrix((weights, (y_true_evt, y_pred_evt)),
                    shape=(n_labels, n_labels)
                    ).toarray()

    # This is old:
    # def get_weighted_unique_CM_entry(row, column):
    #     mask = np.logical_and(
    #         y_true == row,
    #         y_pred == column)

    #     masked_ids = ids[mask]
    #     u_val, u_ind = np.unique(masked_ids, return_index=True)

    #     CM_entry = np.sum(weights[u_ind])
    #     return CM_entry

    # CM = np.zeros((n_labels, n_labels))
    # for irow in range(n_labels):
    #     for icol in range(n_labels):
    #         CM[irow, icol] = get_weighted_unique_CM_entry(irow, icol)

    return CM
