#!/usr/bin/env python
# Final random forest
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report

verbose = False

# Final parameters (selected by random param search): 
# 'n_estimators': 390, 'max_features': 0.15

# Get data in
rawData = pd.read_csv('../../data/toyDataset/toyDataset.csv', sep=';', index_col=0)    # Import data
filteredColumns = pd.read_csv('../filtered.csv')                                       # Import filtered columns
data, labels = rawData[filteredColumns.columns.tolist()], rawData['label']          # Select filtered data
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.1)  # Split data

# Create random forest, fit data
clf = RandomForestClassifier(verbose = False, n_jobs = 20, max_depth = None, n_estimators = 390, max_features = 0.15)
if verbose:
    print 'Fitting data'
clf = clf.fit(trainData, trainLabels)

# Calculate AUC on training data
if verbose:
    print 'Calculating mean training data AUC'
currScores = cross_val_score(clf, trainData, trainLabels, cv = 9, scoring = 'roc_auc')
print 'Mean training AUC:', currScores.mean() 

# Calculate AUC for test data
testPreds = clf.predict(testData)
print "Test AUC:", roc_auc_score(testLabels, testPreds)

if verbose:
    print
    print classification_report(testLabels, testPreds)
