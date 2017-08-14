#!/usr/bin/env python
# Final random forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.noise import AlphaDropout
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.callbacks import TensorBoard

dr = 0.1 # Dropout rate
lr = 0.0001 # Learning rate

# Get data in
raw_data = pd.read_csv('/fhgfs/users/wmartin/toyDataset/toyDataset.csv', sep=';', index_col=0)    # Import data
filtered_columns = pd.read_csv('../filtered.csv')                                       # Import filtered columns

# Select filtered data, scale to have mean 0 and stdev 1
data = StandardScaler().fit_transform(raw_data[filtered_columns.columns.tolist()])

# Convert labels from binary classes to one-hot vector
labels = to_categorical(raw_data['label'])

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.1)

# Prepare TensorBoard
tb = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True)

model = Sequential()

model.add(Dense(64, input_shape=(169,), activation='selu'))
model.add(AlphaDropout(dr))
model.add(Dense(128, activation='selu'))
model.add(AlphaDropout(dr))
# model.add(Dense(128, activation='selu'))
# model.add(Dense(256, activation='selu'))
# model.add(Dense(256, activation='selu'))
# model.add(Dense(128, activation='selu'))
model.add(Dense(128, activation='selu'))
model.add(AlphaDropout(dr))
model.add(Dense(64, activation='selu'))
model.add(AlphaDropout(dr))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=200, batch_size=128, verbose=1, validation_split=0.1, shuffle=True, callbacks=[tb])

model.evaluate(test_data, test_labels)
test_preds = model.predict(test_data)
print "Test AUC:", roc_auc_score(test_labels, test_preds)
