'''
Functions for preparing a waveform dataset for a neural net.
'''

from __future__ import division, print_function

import numpy as np
import pandas as pd
import collections
import sys
import os
import glob
import cPickle as pickle
import tables
from tqdm import tqdm


Datasets = collections.namedtuple('Datasets', ['train', 'val', 'test'])


class DataSet(object):

    def __init__(self,
                 waveforms,
                 labels,
                 weights,
                 ids):
        '''
        Construct a DataSet.
        '''

        assert waveforms.shape[0] == labels.shape[0], (
            'waveforms.shape: {}, labels.shape: {}'.format(
                waveforms.shape, labels.shape))

        assert waveforms.shape[0] == weights.shape[0], (
            'waveforms.shape: {}, weights.shape: {}'.format(
                waveforms.shape, weights.shape))

        self._num_examples = waveforms.shape[0]

        assert waveforms.shape[1] == 128, (
            'waveforms.shape: {}'.format(
                waveforms.shape))

        self._waveforms = waveforms
        self._means = np.mean(self._waveforms, axis=0)
        self._labels = labels
        self._weights = weights
        self._ids = ids
        self._epochs_completed = 0
        self._index_in_epoch = 0

        # New stuff - to only recalculate indices after shuffling
        self._shuffled = True
        self._dp_indices = None
        self._other_indices = None

    @property
    def waveforms(self):
        return self._waveforms

    @property
    def labels(self):
        return self._labels

    @property
    def weights(self):
        return self._weights

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def means(self):
        return self._means

    @property
    def ids(self):
        return self._ids

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def __add__(self, other):
        '''
        Two DataSet objects can now be added with `+`.
        '''
        tot_waveforms = np.append(self._waveforms, other._waveforms, axis=0)
        tot_labels = np.append(self._labels, other._labels, axis=0)
        tot_weights = np.append(self._weights, other._weights, axis=0)
        tot_ids = np.append(self._ids, other._ids, axis=0)
        return DataSet(tot_waveforms, tot_labels, tot_weights, tot_ids)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def adjust_waveforms(self, values):
        assert values.shape[0] == self._waveforms.shape[1], (
            'values.shape: {}, waveforms.shape: {}'.format(
                values.shape, self._waveforms.shape))
        self._waveforms -= values

    def shuffle(self):
        self._shuffled = True
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._waveforms = self._waveforms[perm]
        self._labels = self._labels[perm]
        self._weights = self._weights[perm]
        self._ids = self._ids[perm]

    def next_batch(self, batch_size):
        '''
        Returns the next `batch_size` examples from this dataset.
        '''
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle data
            self.shuffle()

            # Initialize the next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, (
                'batch_size: {}, num_examples: {}'.format(
                    batch_size, self._num_examples))
        end = self._index_in_epoch

        return self._waveforms[start:end], self._labels[start:end]

    # def sort_by_label(self):
    #     '''
    #     Sorts the dataset (for use with the balanced batch method)
    #     '''
    # 
    # The above is for an alternative form of balanced batch which keeps track of the epochs
    # To implement this, perhaps have private variables _sorted, _dpIndex, _dpEpochs, _otherEpochs
    # Scrap above proposed method
    # Reimplement next_batch_balanced to:
    #   - sort by class if !_sorted, set _dpIndex = first index of dp waveform
    #   - use binomial distrib to generate dpBatchSize, otherBatchSize
    #   - proceed as before but with two separate shuffles, one for dp and the other for other
    #   - combine both and shuffle final result
    # Implement shuffleDP, shuffleOther (assert _sorted)
    # Reimplement shuffle (set _sorted = False)
    # Reimplement next_batch (set _sorted = False, shuffle)


    def next_batch_balanced(self, batch_size, dp_prob=0.5):
        '''
        Returns the next `batch_size` examples from this dataset, 
        balanced to have a custom distribution between classes.
        (Allows for undersampling of other waveforms)

        N.B. epoch becomes meaningless using this method
        '''

        # Calculate double pulse and other batch sizes
        dp_batch_size = np.random.binomial(batch_size, dp_prob)
        other_batch_size = batch_size - dp_batch_size

        # If shuffled since last time, recalculate indices
        if self._shuffled:
            self._dp_indices = (self._labels[:, 1] == 1)
            self._other_indices = (self._labels[:, 0] == 1)
            self._shuffled = False

        assert batch_size <= self._dp_indices.shape[0], \
            ('Batch size larger than number of dp waveforms')
        assert batch_size <= self._other_indices.shape[0], \
            ('Batch size larger than number of other waveforms')

        # Generate list of all possible indices
        indices = np.arange(len(self._waveforms))

        # Select samples without replacement
        # Use _dpIndices as mask to select values from indices w/ dp waveforms
        dp_batch_indices = np.random.choice(
                indices[self._dp_indices], 
                dp_batch_size, 
                replace=False
            )

        other_batch_indices = np.random.choice(
                indices[self._other_indices], 
                other_batch_size, 
                replace=False
            )

        # Concatenate to generate batch
        batch_indices = np.concatenate((dp_batch_indices, other_batch_indices))

        # Shuffle batch
        np.random.shuffle(batch_indices)

        # Return waveforms, labels for batch
        return self._waveforms[batch_indices], self._labels[batch_indices]


def get_values_from_table(table, cols, dtype=float):
    values = np.empty((table.nrows, len(cols)), dtype=float)
    for i, row in enumerate(table.iterrows()):
        values[i, :] = [row[col] for col in cols]
    return values


def get_number_of_rows(files, verbose):
    # Swap out tqdm if not verbose
    def verbose_passthrough(a):
        return a

    fn = None
    if verbose:
        fn = tqdm
    else:
        fn = verbose_passthrough

    n_wfs = 0
    for filename in fn(files):
        f = tables.open_file(filename)
        waveform = f.get_node('/waveforms')
        n_wfs += waveform.nrows
        f.close()
    return n_wfs


def get_indices(num_examples, train_ratio=.8, test_ratio=.13):
    # TODO: This should produce a more intuitive result LUL
    # Returns three numbers:
    #  - train_index = max index for training data (i.e. td = data[:train_index])
    #  - test index = no of testing samples (testData = data[train_index:train_index + test_index])
    #  - val_index = start index for validation data (vd = data[val_index])
    train_index = int(num_examples * train_ratio)
    test_index = int(num_examples * test_ratio)
    val_index = train_index + test_index
    return train_index, test_index, val_index


def get_signal_mask(df):
    masks = []
    # Select CC nutau interactions in the detector
    masks.append(df['InIceLabel'] == 161)
    # Select waveforms whose DOMs are close to both interaction
    # vertices and still suficiently separated in time
    masks.append(df['GeometricalSelection'] == 1)
    # Apply a light set of cuts on the remaining waveforms
    masks.append(df['Bins_ToT_Pulse1'] >= 2)
    masks.append(df['Bins_ToT_Pulse2'] >= 3)
    masks.append(df['Bins_TbT'] >= 2)
    masks.append(df['Amplitude_Pulse1'] >= 10)
    masks.append(df['Amplitude_Pulse2'] >= 10)

    # Combine all the masks
    selection_mask = masks[0]
    for i in range(1, len(masks)):
        selection_mask = np.logical_and(selection_mask, masks[i])

    return selection_mask


def read_data(datasets, combined=False, weight_name='hese_flux', train_ratio=.8, test_ratio=.13, verbose=True):
    '''
    Reads data. `datasets` should be a list of tuples. Each tuple has to
    contain the datasetnumbers as a string, the interactions to be
    read in ('NC' and/or 'CC'), as well as the number of files in Madison
    to adjust the weights.

    Return a list of DataSets for each element in `datasets`. If `combined`
    all elements get added and one DataSet will be returned.

    The `weight_name` corresponds to the name of the wanted weight in the
    hdf5 files.
    '''
    def verbose_passthrough(a):
        return a

    for dset in datasets:
        assert len(dset) == 3, (
            'each dataset should contain the datasetnumber '
            'the interactions and '
            'the number of files in Madison')

    if os.environ['LOGNAME'] == 'wmartin':
        path = '/fhgfs/users/wmartin/waveformData/{}/*/*.hd5'
    else:
        print('Set up path correctly for your system!')
        sys.exit(1)

    if not isinstance(datasets, list):
        datasets = [datasets]
    DSlist = []
    for dset in datasets:
        files = glob.glob(path.format(dset[0]))

        n_wfs = get_number_of_rows(files, verbose)

        wfs = np.empty((n_wfs, 128))
        interaction = np.empty((n_wfs, 1))
        weights = np.empty((n_wfs, 1))
        id_values = np.empty((n_wfs, 3))

        # Extract derivate informations only for the nutau dataset
        # to define the signal class
        if dset[0] == '11538':
            feature_list = ['Bins_ToT_Pulse1', 'Bins_ToT_Pulse2',
                            'Bins_TbT', 'Amplitude_Pulse1',
                            'Amplitude_Pulse2', 'GeometricalSelection',
                            'InIceLabel']
            derivate_values = np.empty((n_wfs, len(feature_list)))

        from_i = 0

        fn = None
        if verbose:
            fn = tqdm
        else:
            fn = verbose_passthrough

        for filename in fn(files):
            f = tables.open_file(filename)
            data = f.get_node('/data')
            weight_node = f.get_node('/weights')
            waveform = f.get_node('/waveforms')

            rows = waveform.nrows
            to_i = from_i + rows

            interaction[from_i:to_i] = get_values_from_table(
                data, ['interaction'])
            weights[from_i:to_i] = get_values_from_table(
                weight_node, [weight_name])
            id_values[from_i:to_i] = get_values_from_table(
                data, ['run_id', 'event_id', 'sub_event_id'])
            if dset[0] == '11538':
                derivate_values[from_i:to_i] = get_values_from_table(
                    data, feature_list)
            wfs[from_i:to_i] = f.root.waveforms[:]
            from_i += rows
            f.close()

        if dset[0] == '11538':
            derivate_df = pd.DataFrame(columns=feature_list,
                                       index=np.arange(n_wfs))
            derivate_df[feature_list] = derivate_values

        interaction = interaction.flatten()
        weight = weights.flatten()

        # create a unique ID for each event
        unique_id = np.int64(id_values[:, 0] * 1e5 +
                             id_values[:, 1] * 10 +
                             id_values[:, 2]).flatten()
        n_files = len(files)
        n_files_mad = dset[2]

        weight *= (n_files_mad / n_files)

        for comp in dset[1]:
            if dset[0] == '11538' and comp == 'DP':
                # Apply cuts on the derivate df
                signal_mask = get_signal_mask(derivate_df)
                comp_wfs = wfs[signal_mask, :]
                weights = weight[signal_mask]
                ids = unique_id[signal_mask]
                lbl = 1
            elif comp == 'NC':
                mask = (interaction == 0)
                comp_wfs = wfs[mask, :]
                weights = weight[mask]
                ids = unique_id[mask]
                lbl = 0
            elif comp == 'CC':
                if dset[0] == '11538':
                    raise NotImplementedError(
                        'This would assign Label 0 to all nutau CC events')
                mask = (interaction == 1)
                comp_wfs = wfs[mask, :]
                weights = weight[mask]
                ids = unique_id[mask]
                lbl = 0

            def labels(shape, lbl, n_classes=2):
                labels = np.zeros((shape, n_classes))
                labels[:, lbl] = 1
                return labels

            num_examples = comp_wfs.shape[0]
            train_index, test_index, val_index = get_indices(num_examples, train_ratio = train_ratio, test_ratio = test_ratio)
            train = DataSet(comp_wfs[:train_index],
                            labels(train_index, lbl),
                            weights[:train_index],
                            ids[:train_index])
            test = DataSet(comp_wfs[train_index:train_index+test_index],
                           labels(test_index, lbl),
                           weights[train_index:train_index+test_index],
                           ids[train_index:train_index+test_index])
            val = DataSet(comp_wfs[val_index:],
                          labels(num_examples-val_index, lbl),
                          weights[val_index:],
                          ids[val_index:])
            DS = Datasets(train=train, test=test, val=val)
            DSlist.append(DS)

    if combined:
        DS = Datasets(train=sum([ds.train for ds in DSlist]),
                      val=sum([ds.val for ds in DSlist]),
                      test=sum([ds.test for ds in DSlist]))
        # At least free up some memory
        del DSlist
        DS.train.shuffle()
        DS.val.shuffle()
        DS.test.shuffle()
        # # Mean subtraction
        # means = DS.train.means
        # DS.train.adjust_waveforms(means)
        # DS.val.adjust_waveforms(means)
        # DS.test.adjust_waveforms(means)
        return DS
    else:
        print('No mean subtraction done yet!')
        return DSlist


if __name__ == '__main__':
    # ## Example usage ## #
    datasets = [
        ('11538', ['DP', 'NC'], 3638),  # Tau
        ('12034', ['CC', 'NC'], 8637),  # Electron
        ('11069', ['NC'], 7287)         # Muon (discard muon track events because of similarity to dp events)
    ]
    ds_list = read_data(datasets)
