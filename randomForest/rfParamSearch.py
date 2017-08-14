#!/usr/bin/env python
# Random forest with a parameter search to maximise the AUC
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report

# Coarse search with 100 iterations:
# 'n_estimators': randLogInt(1, 3), 'max_features': scipy.stats.uniform(), test_size = 0.9, cv = 3
# Best params: {'max_features': 0.068586334184792985, 'n_estimators': 340}
# 
# Medium search with 2x20 iterations:
# 'n_estimators': randLogInt(2.3, 2.7), 'max_features': scipy.stats.uniform(loc = 0.05, scale=0.15), test_size = 0.9, 0.5, cv = 3
# Best params (test_size = 0.9): {'max_features': 0.12843108845096549, 'n_estimators': 399}
# Best params (test_size = 0.5): {'max_features': 0.15189851588709988, 'n_estimators': 367}
#
# Fine search with 20 iterations:
# 'n_estimators': randLogInt(2.5, 2.6), 'max_features': scipy.stats.uniform(loc = 0.05, scale=0.15), test_size = 0.1, cv = 9
# Best params: {'max_features': 0.143870664725816, 'n_estimators': 329}
#
# Fine search with 100 iterations:
# 'n_estimators': randLogInt(2.5, 2.6), 'max_features': scipy.stats.uniform(loc = 0.05, scale=0.15), test_size = 0.1, cv = 9
# Best params: {'max_features': 0.14782535876743441, 'n_estimators': 386}

# Final parameters: 
# 'n_estimators': 390, 'max_features': 0.15

# Coarse search yielded the following results: 
#   -  0.05 - 0.20 for max_features (in range of sqrt(169))
#   -  200 - 400 for n_estimators

class randLogInt:
    ''' Generates random integers evenly distributed in log space between low and high '''
    def __init__(self, low, high):
        self.x = scipy.stats.uniform(loc = low, scale = (high - low))
        
    def rvs(self, size=-1, random_state=None):
        if size == -1:
            return (10**self.x.rvs(random_state=random_state)).astype(int)
        else:
            return (10**self.x.rvs(size = size, random_state=random_state)).astype(int)

rawData = pd.read_csv('../../data/toyDataset/toyDataset.csv', sep=';', index_col=0)    # Import data
filteredColumns = pd.read_csv('../filtered.csv')                                       # Import filtered columns
data, labels = rawData[filteredColumns.columns.tolist()], rawData['label']          # Select filtered data
trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.1)  # Split data

# Conduct randomised X-Val search
tunedParams = {'n_estimators': randLogInt(2.5, 2.6), 'max_features': scipy.stats.uniform(loc = 0.05, scale=0.15)}
clf = RandomizedSearchCV(RandomForestClassifier(verbose = True, n_jobs = 20, max_depth = None), tunedParams, n_iter = 100, cv = 9, scoring = 'roc_auc')
clf = clf.fit(trainData, trainLabels)

print "Best params: "
print clf.best_params_

print 'Score: '
currScores = cross_val_score(clf, trainData, trainLabels)
print currScores.mean() 

# Test time
testPreds = clf.predict(testData)
print "AUC:", roc_auc_score(testLabels, testPreds)
print
print classification_report(testLabels, testPreds)

scatter = plt.scatter(clf.cv_results_['param_max_features'].data, clf.cv_results_['param_n_estimators'].data, c= clf.cv_results_['mean_test_score'])
cb = plt.colorbar(scatter)
#plt.xlim([0, 1])
#plt.ylim([0, 1000])
plt.title("Fine Random Parameter Search")
plt.xlabel('max_features')
plt.ylabel("n_estimators")
cb.set_label('Mean AUC on Training Data')
plt.show()
