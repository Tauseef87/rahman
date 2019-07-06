import pandas as pd
from sklearn import naive_bayes
from sklearn import preprocessing,model_selection
from sklearn_pandas import  CategoricalImputer
import seaborn as sns

#creation of data frames from csv
titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')
print(titanic_train.info())

#preprocessing Stage
#impute missing values for continoius feature
imputable_cont_feature = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_feature])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_feature]=cont_imputer.transform(titanic_train[imputable_cont_feature])

#impute missing values for categorical feature
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked']=cat_imputer.transform(titanic_train['Embarked'])

le_embarked = preprocessing.LabelEncoder()
le_embarked.fit(titanic_train['Embarked'])
titanic_train['Embarked']=le_embarked.transform(titanic_train['Embarked'])


le_sex = preprocessing.LabelEncoder()
le_embarked.fit(titanic_train['Sex'])
titanic_train['Sex']=le_embarked.transform(titanic_train['Sex'])

features = ['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']
X_train=titanic_train[features]
Y_train = titanic_train['Survived']

sns.distplot(X_train['Fare'],hist=False)
sns.distplot(X_train['Age'],hist=False)
sns.distplot(X_train['Pclass'], hist=False)
sns.distplot(X_train['Sex'], hist=False)

gnb_estimator = naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
gnb_estimator.fit(X_train,Y_train)
print(gnb_estimator.class_prior_) # Probality of classes
print(gnb_estimator.sigma_) #Variance in GaussianNB algo
print(gnb_estimator.theta_) #Mean in GaussianNB algo

scores = model_selection.cross_validate(gnb_estimator,X_train,Y_train,cv=10)
test_scores = scores.get("test_score")
train_score = scores.get("train_score")
print(test_scores)
print(train_score)

#read test data
titanic_test=pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\test.csv')
print(titanic_test.info())
titanic_test[imputable_cont_feature]=cont_imputer.transform(titanic_test[imputable_cont_feature])
cat_imputer.fit(titanic_test['Embarked'])
titanic_test['Embarked']=cat_imputer.transform(titanic_test['Embarked'])
le_embarked.fit(titanic_test['Embarked'])
titanic_train['Embarked']=le_embarked.transform(titanic_train['Embarked'])
