import sys
sys.path.append('C://Users//tauseef.ur.rahman//Desktop//MyPythonfiles')

import classification_utils as cutils
from sklearn import model_selection, tree
import pydot # This module provides with a full interface to create handle modify and process graphs in Graphviz's dot language
'''
Graphviz's dot language - DOT can be used to describe an undirected graph. An undirected graph shows simple relations between objects
, such as friendship between people. The graph keyword is used to begin a new graph, and nodes are described within curly braces
. A double-hyphen (--) is used to show relations between the nodes.
'''
import io
import os
import pandas as pd
import numpy as np

X_train,Y_train = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=500,noise=0.30) 
#generate_nonlinear_synthetic_data_classification3 -> for make_moons
cutils.plot_data_2d_classification(X_train,Y_train)

#underfitted learning in dt
dt_estimator = tree.DecisionTreeClassifier(max_depth=1)
dt_estimator.fit(X_train, Y_train)
cv_scores = model_selection.cross_val_score(dt_estimator, X_train, Y_train, cv= 10)
print(np.mean(cv_scores))
train_score = dt_estimator.score(X_train, Y_train)
print(train_score)

#overfitted learning in dt
dt_estimator = tree.DecisionTreeClassifier(max_depth=15)
dt_estimator.fit(X_train, Y_train)
cv_scores = model_selection.cross_val_score(dt_estimator, X_train, Y_train, cv= 10)
print(np.mean(cv_scores))
train_score = dt_estimator.score(X_train, Y_train)
print(train_score)
