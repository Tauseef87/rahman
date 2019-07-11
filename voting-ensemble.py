import sys
sys.path.append('C://Users//tauseef.ur.rahman//Desktop//MyPythonfiles')

import classification_utils as cutils
from sklearn import model_selection,ensemble,tree,neighbors
import numpy as np

#2-d classification
X,y = cutils.generate_linear_synthetic_data_classification(n_samples=1000,n_features=10,n_classes=2,weights=[0.5,0.5])
cutils.plot_data_3d_classification(X,y)
cutils.plot_data_2d_classification(X,y)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2,random_state=1)

ax = cutils.plot_data_2d_classification(X_train, y_train)
ax = cutils.plot_data_2d_classification(X_test, y_test, ax, marker='x', s=70, legend=False)

dt_estimator = tree.DecisionTreeClassifier()
dt_grid = {'criterion':['gini','entropy'],'max_depth':list(range(1,9))}
dt_grid_estimator = model_selection.GridSearchCV(dt_estimator,dt_grid,cv=10,refit=True)
#refit - If you want to use predict, you'll need to set 'refit' to True
dt_grid_estimator.fit(X_train,y_train)

knn_estimator = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':list(range(1,9))}
knn_grid_estimator = model_selection.GridSearchCV(knn_estimator,knn_grid,cv=10,refit=True)
knn_grid_estimator.fit(X_train,y_train)

rf_estimator = ensemble.RandomForestClassifier()
rf_grid = {'n_estimators':list(range(50,100,10)),'max_depth':list(range(5,9))}
rf_grid_estimator = model_selection.GridSearchCV(rf_estimator,rf_grid,cv=10,refit=True)
rf_grid_estimator.fit(X_train,y_train)

#hard voting
hvoting_estimator = ensemble.VotingClassifier([('dt',dt_estimator),('knn',knn_estimator),('rf',rf_estimator)])
hvoting_estimator.fit(X_train,y_train)
cv_scores = model_selection.cross_val_score(hvoting_estimator,X_train,y_train)
#cross_val_score - Evaluate a score by cross-validation
print(np.mean(cv_scores))

#soft voting
svoting_estimator = ensemble.VotingClassifier([('dt',dt_estimator),('knn',knn_estimator),('rf',rf_estimator)],voting='soft')
svoting_estimator.fit(X_train,y_train)
cv_scores = model_selection.cross_val_score(svoting_estimator,X_train,y_train)
print(np.mean(cv_scores))

#with weight
svoting_estimator = ensemble.VotingClassifier([('dt',dt_estimator),('knn',knn_estimator),('rf',rf_estimator)],voting='soft',weights=[1,1,3])
svoting_estimator.fit(X_train,y_train)
cv_scores = model_selection.cross_val_score(svoting_estimator,X_train,y_train)
print(np.mean(cv_scores))

#Grid Search
voting_estimator = ensemble.VotingClassifier([('dt',dt_estimator),('knn',knn_estimator),('rf',rf_estimator)])
voting_grid = {'voting':['hard','soft'],'weights':[(1,1,1) , (1,1,2), (1,1,3)]}
voting_grid_estimator = model_selection.GridSearchCV(voting_estimator,voting_grid,cv=10,refit=True)
voting_grid_estimator.fit(X_train,y_train)
print(voting_grid_estimator.best_estimator_)
print(voting_grid_estimator.best_params_)
print(voting_grid_estimator.best_score_)





