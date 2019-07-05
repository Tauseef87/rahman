import sys 
'''
This module provides access to some variables used or maintained by the interpreter and to 
functions that interact strongly with the interpreter. It is always available.
'''
sys.path.append('C://Users//tauseef.ur.rahman//Desktop//MyPythonfiles')
#Sys.path is python installation path and append will add new path 

import classification_utils as cutils
from sklearn import model_selection,metrics,neighbors 
import numpy as np

X,y = cutils.generate_linear_synthetic_data_classification(n_samples=1000,n_features=2,n_classes=4,weights=[0.3,0.3,0.3,0.3])
#make_classification X : array of shape [n_samples, n_features] it will features and its values,
#y : array of shape [n_samples]  The integer value or labels for class membership of each sample
cutils.plot_data_2d_classification(X,y)

X_train,X_text,Y_train,Y_test = model_selection.train_test_split(X,y,test_size = 0.2,random_state=1)
# model_selection.train_test_split - Split arrays or matrices into random train and test subsets
#test_size - represent the proportion of the dataset to include in the test split
# random_state - If int, random_state is the seed used by the random number generator

knn_estimator = neighbors.KNeighborsClassifier()
knn_estimator.fit(X_train,Y_train)

cv_scores = model_selection.cross_val_score(knn_estimator, X_train, Y_train, cv = 10)
#model_selection.cross_val_score - Evaluate a score by cross-validation