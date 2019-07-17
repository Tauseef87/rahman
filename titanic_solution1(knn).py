import sys
sys.path.append('C://Users//tauseef.ur.rahman//Desktop//MyPythonfiles')

import pandas as pd
import os
import common_utils as cutils
from sklearn import preprocessing,neighbors,svm,linear_model,ensemble,pipeline
import classification_utils as clutils
dir = 'C://Users//tauseef.ur.rahman//Desktop//Python-Docs//Algorithmica_Pdf'
titanic_train = pd.read_csv(os.path.join(dir,'train.csv'))

print(titanic_train.shape)
print(titanic_train.info())

titanic_train1 = cutils.drop_features(titanic_train,['PassengerId', 'Name', 'Survived', 'Ticket', 'Cabin'])
# drop_feature = dataframe.drop Drop specified labels from rows or columns
#axis =Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’)

#type casting.cast_cont_to_cat
cutils.cast_to_cat(titanic_train1,['Sex','Embarked','Pclass'])
#DataFrame.astype - to cast one or more of the DataFrame’s columns to column-specific types

cat_features = cutils.get_categorical_features(titanic_train1)
#DataFrame.select_dtypes - Return a subset of the DataFrame’s columns based on the column dtypes.
#include, exclude- A selection of dtypes or strings to be included/excluded - 'category'
print(cat_features)

cont_features = cutils.get_continuous_features(titanic_train1)
#DataFrame.select_dtypes - Return a subset of the DataFrame’s columns based on the column dtypes.
#include, exclude- A selection of dtypes or strings to be included/excluded - 'number'
print(cont_features)

#handle missing data(imputation)
cat_imputers = cutils.get_categorical_imputers(titanic_train1,cat_features)
#DataFrameMapper, a class for mapping pandas data frame columns to different sklearn transformations
#By default the transformers are passed a numpy array of the selected columns as input
#This is because sklearn transformers are historically designed to work with numpy arrays
#we can pass a dataframe/series to the transformers to handle custom cases initializing the dataframe mapper with input_df=True
#By default the output of the dataframe mapper is a numpy array. This is so because most sklearn estimators expect a numpy array as input
#If however we want the output of the mapper to be a dataframe, we can do so using the parameter df_out
titanic_train1[cat_features] = cat_imputers.transform(titanic_train1[cat_features])

con_imputers = cutils.get_continuous_imputers(titanic_train1,cont_features)
titanic_train1[cont_features] = con_imputers.transform(titanic_train1[cont_features])

#adding new levels
#titanic_train['Pclass'] = titanic_train['Pclass'].cat.add_categories([4,5])

#one -hot encoding
#One hot encoding is used when there exists no ordinal relationship in column
#Ordinal variables are variables that are categorized in an ordered format, so that the different categories can be ranked 
#from smallest to largest or from less to more on a particular characteristic
X_train = cutils.ohe(titanic_train1,cat_features)
#get_dummies Convert categorical variable into dummy/indicator variables
#A dummy variable (aka, an indicator variable) is a numeric variable that represents categorical data, such as gender, race, political affiliation, etc.
y_train = titanic_train['Survived']

#build model
knn_pipelines_stages = [
                       ('scaler', preprocessing.StandardScaler()),
                       ('knn', neighbors.KNeighborsClassifier())
                       ]
knn_pipeline = pipeline.Pipeline(knn_pipelines_stages)
knn_pipeline_grid = {'knn__n_neighbors':list(range(1,10))}
knn_pipeline_model = cutils.grid_search_best_model(knn_pipeline,knn_pipeline_grid,X_train,y_train)

titanic_test=pd.read_csv(os.path.join(dir,'test.csv'))
titanic_test1 = cutils.drop_features(titanic_test,['PassengerId', 'Name', 'Ticket', 'Cabin'])
cutils.cast_to_cat(titanic_test1,['Sex','Embarked','Pclass'])
cont_features = cutils.get_continuous_features(titanic_test1)
cat_features = cutils.get_categorical_features(titanic_test1)
titanic_test1[cat_features] = cat_imputers.transform(titanic_test1[cat_features])
titanic_test1[cont_features] = con_imputers.transform(titanic_test1[cont_features])
X_test = cutils.ohe(titanic_test1,cat_features)
titanic_test['Survived'] = knn_pipeline_model.predict(X_test)
