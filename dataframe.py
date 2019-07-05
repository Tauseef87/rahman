import pandas as pd
import os
titanic_train = pd.read_csv('C:\\Users\\tauseef.ur.rahman\\Desktop\\Python-Docs\\Titanic\\train.csv')
person = {'name':['abc','xyz'],'age':[10,20],'fare':[12.4,13.5]}
df = pd.DataFrame(person)
print(df)
print(df.shape)
print(df.info())
print(df.index)
print(df.values)
df = df.set_index(df.age)
print(df)
df = df.reset_index()
df = df.reset_index(drop=True)

print(titanic_train.shape)
print(titanic_train.info())
print(titanic_train.columns)
print(titanic_train.dtypes) # datatypes of each column
print(titanic_train.values)
print(titanic_train.sample(n=4)) #  sample gives random rows and column based on n. where n specifies no.of rows
print(titanic_train.sample(frac = 0.1)) # frac denotes 10% of random values from data frame

#row and column access based on index
titanic_train.iloc[1:3] #iloc gets rows (or columns) at particular positions in the index (so it only takes integers).
titanic_train.iloc[0:3,0:2] # first two columns of data frame with first 3 rows

#row and column access based on name
titanic_train.loc[1:3, ['Sex','Fare']] #first two columns with specific column name with first 3 rows

#axis=0 means row dimension
#axis=1 means column dimension
#filter rows or columns based on labels

titanic_train.filter(items=['Age', 'Fare'], axis=1).head(3) # filter based on column name
titanic_train.filter(items=[5,10,12], axis=0).head() #filter based on rows numnber

#sorting dataframe
titanic_train.sort_index(ascending = False).head(3) # sort the dataframe in descending order and give top 3 rows
titanic_train1 = titanic_train.sort_values(by=['Fare','Age'],ascending = [False,True]).head()
# or
titanic_train1 = titanic_train.filter(items=['Fare','Age']).sort_values(by=['Fare','Age'],ascending = [False,True])
titanic_train1

#group by
g = titanic_train.groupby('Sex').size() # group by sex column size -Return the number of rows if Series. Otherwise return the number of rows times number of columns if DataFrame.
print(g.mean)









