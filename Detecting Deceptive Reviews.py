'''
Reviews intended to promote or hype an offering, and which therefore express a positive sentiment 
towards the offering, are called positive deceptive opinion spam.

In contrast, reviews intended to disparage or slander(Badnaam) competitive offerings, and which therefore 
express a negative sentiment towards the offering, are called negative deceptive opinion spam.

In this Data we distinguish between two kinds of deceptive opinion spam, 
depending on the sentiment expressed in the review i.e i.e positive Deceptive and Negative Deceptive Review

We will build a end to end solution which will help to pass a review and will give you back a label 
deciding whether the review is Deceptive or Not

'''

import pandas as pd
pd.set_option('display.max_columns',100) # Display max column in dataframe
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') #Remove warning from notebook

import os
#change the working directory to Deceptive Dataset
os.chdir('F:/DeceptiveDataset')


#Collecting Data

import zipfile
with zipfile.ZipFile('op_spam_v1.4.zip','r') as z:
    z.extractall()
    
#Importing Data
#The following function gives a tree structure of the files inside op_spam_v1.4/ folder

def file_list(startpath):
 for root,dirs,files in os.walk(startpath): 
     #OS.walk() generate the file names in a directory tree by walking the tree either top-down or bottom-up
     #root : Prints out directories only from what you specified.
     #dirs : Prints out sub-directories from root.
     #files : Prints out all files from root and directories.
     
     level = root.replace(startpath,'').count(os.sep)
     #In this code we are counting seperater like '/' by replacing op_spam_v1.4
     # from full root of files i.e op_spam_v1.4/negative_polarity
     indent = ' '*6*(level)
     print('{}{}/'.format(indent,os.path.basename(root)))
     #os.path.basename() method in Python is used to get the base name in specified path
     subindent = ' '*6*(level+1)
     for f in files[:5]:
         print('{}{}'.format(subindent,f))
         
 file_list('op_spam_v1.4/')        
 
 #op_spam_v1.4/ folder has two sub-folders negative_polarity/ and positive_polarity/
 #Now Each polarity folder has two sub-folder deceptive_from_MTurk/ and truthful_from_TripAdvisor/
 #Directories prefixed with fold correspond to a single fold from the cross-validation experiments.
 #Files are named according to the format %c%h%i.txt, where
 #%c denotes the class: (t)ruthful or (d)eceptive
 #%h denotes the hotel
 #%i serves as a counter to make the filename unique
 
 #Reading the files in a Dataframe
 
 #Reading Positive reviews files inside positive_polarity/.
 
 #Accesing data for positive reviews
 
import glob
import ntpath

#The ntpath module provides os.path functionality on Windows platforms.
ntpath.basename('a/b/c/abc.txt') # Example of ntpath
#It returns the filename from the whole path

#You can also use it to handle Windows paths on other platforms.

path = 'op_spam_v1.4/positive_polarity/'
files = [f for f in glob.glob(path+'**/*.txt',recursive=True)]
#glob searches for files with the regex \*\*/\*.txt inside folders and sub-folders.
len(files)

#There are total 800 reviews in positive sentiment folder, out of which 400 are true and 400 are deceptive.

#Now we iterate over each filename in list and read them into a DataFrame.

filename = list()
reviews =list()

for file in files:
    with open(file,'r') as f:
        filename.append(ntpath.basename(file))
        reviews.append(f.read())
        
positive_df = pd.DataFrame({'filename':filename,'reviews':reviews})        
positive_df.head()
positive_df.shape

#As we are done reading all the positive files in DataFrame
#Now lets start Reading Negative reviews files inside negative_polarity/
path = 'op_spam_v1.4/negative_polarity/'
files = [f for f in glob.glob(path+'**/*.txt',recursive=True)]
filename = list()
reviews =list()
for file in files:
    with open(file,'r') as f:
        filename.append(ntpath.basename(file))
        reviews.append(f.read())

negative_df = pd.DataFrame({'filename':filename,'reviews':reviews})        
negative_df.head()
negative_df.shape

#Data Preparation
#Our very First step in Data Preparation is to merge positive and negative DataFrames and assigning 
#a label to it specifying whether the review is positive or negative.

positive_df['polarity']= ['positive'] * positive_df.shape[0]
negative_df['polarity']= ['negative'] * positive_df.shape[0]

final_df = pd.concat([positive_df,negative_df],axis=0,ignore_index=True)
#Axis - Axis to concate along , 0=Index,1=column

#Next Step in Data Preparation is to extract hotel name and our target label whether the 
#review is true or deceptive.All these column lies in our filenames and we have to use some regular 
#expression and extract it

final_df['Hotel']=final_df.filename.str.extract('_(.*)_')

#Extract the very first character from the filename and map d to deceptive and t to true.

final_df['label']=final_df.filename.str[0].map({'d':'deceptive','t':'true'})
#Here str[0] means first letter of filename i.e d or t
#map() function returns a map object(which is an iterator) of the results after applying the given 
#function to each item of a given iterable (list, tuple etc.)

#Remove filename from dataframe
final_df.drop('filename',axis=1,inplace=True)

#final_df.to_csv('final_df.csv', index= False)


df= final_df.copy()

#Exploratory Data Analysis

df.groupby(['polarity','label']).count()

#As discussed in the Importing Data section each polarity i.e positive and negative has 400 deceptive and 400 true

#Unique hotel name

df.Hotel.unique()
df.groupby(['Hotel','polarity']).count()

#Comparing word count in True and deceptive review
#To get the word counts in each reviews we need to tokenise the reviews and we are going to use spaCy 

import spacy
nlp= spacy.load('en_core_web_lg')
df['spacy']=df.reviews.str.lower().apply(nlp)

#Now spacy column in DataFrame has spaCy document object which we can use further for our preprocessing.

def word_count(row):
    return len(list(row))

df['word_count']=df.spacy.apply(word_count)

'''
"Its is first spacy object ->my husband and i satayed for two nights at the hilton chicago
,and enjoyed every minute of it! the bedrooms are immaculate,and the linnens are very soft. 
we also appreciated the free wifi,as we could stay in touch with friends while staying in chicago. 
the bathroom was quite spacious,and i loved the smell of the shampoo they provided-not like most hotel shampoos. 
their service was amazing,and we absolutely loved the beautiful indoor pool. 
i would recommend staying here to anyone.
'''

sns.set()
fig = plt.figure(figsize=(19,8))
sns.distplot(df.loc[df.label=='true','word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='TRUE')
sns.distplot(df.loc[df.label=='deceptive','word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='DECEPTIVE')

#Observations:

#There is not much diffrence in distribution of true and deceptive reviews.

#Comparing Punctuation counts in True and Deceptive Review
    

import string
count = lambda l1,l2:len(list(filter(lambda c: c in l2,l1)))
#The filter() method filters the given sequence with the help of a function that tests each element in the sequence 
#to be true or not
#filter(function, sequence)
#function that tests if each element of a  sequence true or not.
#sequence which needs to be filtered, it can  be sets, lists, tuples, or containers of any iterators.
#returns an iterator that is already filtered

#Count is lambda function I creared which will be count occurances of any item of a list
#(In our case list of punctuation) in a string(In our case review).

punc = list(set(string.punctuation))

def punc_count(row):
    return count(row,punc)

df['punc_count']=df.reviews.apply(punc_count)
#Pandas.apply allow the users to pass a function and apply it on every single value of the Pandas series


#Comparing Describe in the true and deceptive reviews
pd.DataFrame({'True':df[df.label=='true']['punc_count'].describe().values,'Deceptive':
             df[df.label=='deceptive']['punc_count'].describe().values},index=df.describe().index)

#Observations:

#Punctuations in True Reviews are a bit more skewed towards the right comapred to the Deceptive.
    
#Comparing the count of people citing hotel names in review

pd.set_option('display.max_rows',1600)   
df.info() 
df['hotel_name_count'] = df.apply(lambda x:str(x.reviews).lower().count(str(x.Hotel)),axis=1)
df.groupby(['label', 'polarity'])['hotel_name_count'].sum()
df.groupby(['label', 'polarity'])['hotel_name_count'].sum().unstack().plot(kind= 'bar', rot= 45, figsize= (12, 7));

#Observations:

#Thats a great insight you can see right away that people writing deceptive reviews cites the names of Hotel 
#a lot in their review.

#Comparing Number of unique words in reviews

df['unique_word_count']=df['spacy'].apply(lambda x :len(set(list(x))))

sns.set()
fig = plt.figure(figsize=(19,8))
sns.distplot(df.loc[df.label=='true','unique_word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='TRUE')
sns.distplot(df.loc[df.label=='deceptive','unique_word_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='DECEPTIVE')

#Observations:

#Same as Word Count there is no difference here as well.

#Comparing number of stop words
print(nlp.Defaults.stop_words)

#We are going to use all the stop words available in spacy Language Model.
import string
count = lambda l1,l2:len(list(filter(lambda c: c in l2,l1)))
stop = list(nlp.Defaults.stop_words)
df['stop_count']=df.reviews.apply(lambda x:count(x,stop))

#We are going to use the same count Function, I used for counting the number of punctuation, 
#but here instead of punctuations we have list of stop words.

sns.set()
fig = plt.figure(figsize=(19,8))
sns.distplot(df.loc[df.label=='true','stop_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='TRUE')
sns.distplot(df.loc[df.label=='deceptive','stop_count'],hist=False,kde=True,kde_kws={'shade':True,'linewidth':3},label='DECEPTIVE')

#Observations

#Not Much Difference in Stop Words



#Counting the POS Tags

#This Below function takes in the DataFrame and returns a dataframe comparing the POS tags counts 
#in the true and deceptive reviews.


from collections import Counter
def pos_tag_count(df):
    
    # Create to empty lists in which we will store the list of POS Tags
    
    dec_pos_tags= list()
    true_pos_tags= list()
    # Iterate over Each Spacy Document Object
    # Below code will take spacy object where label = true
    
    for doc in df[df.label == 'true'].spacy:
        #print(type(doc)) -> doc is an object of spacy
        TAG_count = doc.count_by(spacy.attrs.TAG)
        #count_by -> Count the frequencies of a given attribute,Produces a dict of {attr (int): count (ints)}
        #You can find the number of occurrences of each POS tag by calling the count_by on the spaCy document object
        #TAG_count ->It is a dictionary object.
        true_dict = dict()
        
        # Spacy Returns a dictionary object with Hash Value of Pos and Counts
        # After every Iteration Update the POS Tag distioanry
        
        for key,value in sorted(TAG_count.items()):
            # print(key,' ',value)
             #print(doc.vocab[key].text,value)
           # print(doc.vocab[key].text,value)
             true_dict.update({ doc.vocab[key].text : value})
             #here we are updating a dictionary with Key value of TAG i.e Key=99 and its vocab=SYM 
             #and the count of SYM
            
        # After updating the dict append it to the list we created at the very beiginning
        
        true_pos_tags.append(true_dict)
        #Then we are appending the list with dictionary value
    
    # Then just follow the same above steps for the deceptive reviews as well  
    for doc in df[df.label == 'deceptive'].spacy:
        TAG_count = doc.count_by(spacy.attrs.TAG)
        
        dec_dict = dict()
        for key,value in sorted(TAG_count.items()):
           
            dec_dict.update({ doc.vocab[key].text : value})
        dec_pos_tags.append(dec_dict)
    
    '''    
    NoW the above returns a count object for each review in True and Deceptive we need to add them up
    for whole DataFrame using a counter object
    '''
    true_tag_count = Counter()
    #print(true_tag_count)
    #Counter is a sub-class which is used to count hashable objects
    #It is an unordered collection where elements are stored as dictionary keys and their 
    #counts are stored as dictionary values
    for i in range(0,len(true_pos_tags)):
        #print(Counter(true_pos_tags[i]))
        #print(true_tag_count)
        true_tag_count = true_tag_count + Counter(true_pos_tags[i])
        #Here what counter is doing that it true_pos_tags list and count the occurance of tag i.e 'JJ': 14
        #And put it in another counter object
        #Counter returns a dictionary object so print(true_tag_count['SYM']-> with count of SYM)
    dec_tag_count = Counter()
    for i in range(0,len(dec_pos_tags)):
        dec_tag_count = dec_tag_count + Counter(dec_pos_tags[i])
    true_tag_count = dict(true_tag_count)
    dec_tag_count = dict(dec_tag_count)
    
    ## Now Create a DataFrame for True Reviews with POS Tags and Counts    
    true_pos = pd.DataFrame()
    true_pos['tag'] = true_tag_count.keys()
    true_pos['true_cnt'] = true_tag_count.values()
    
    ## Now Create a DataFrame for Deceptive Reviews with POS Tags and Counts
    dec_pos = pd.DataFrame()
    dec_pos['tag'] = dec_tag_count.keys()
    dec_pos['dec_cnt'] = dec_tag_count.values()
    
    # Merge these two DataFrames on Tag name
    reviews_tags = true_pos.merge(dec_pos, on='tag', how='left')
    #Merge DataFrame or named Series objects with a database-style join
    #on - label or list
    #Column or index level names to join on
    #how{‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
    #left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
    #inner: use intersection of keys from both frames, similar to a SQL inner join;
    reviews_tags = reviews_tags.set_index('tag')
    return reviews_tags


#The above function takes in the DataFrame and returns the count of the POS tags in true and deceptive reviews setting the index to the POS Tags.
    
pos_tag_count(df).head(10)    
    
#POS Tags Counts in Positive Reviews¶
fig, ax = plt.subplots(figsize=(20, 25))
pos_tag_count(df[df.polarity=='positive']).sort_values(by=['true_cnt']).plot(kind='barh',rot=0,ax=ax)
#sort_values -> Sort by the values along either axis. by = Name or list of names to sort by
#plot - > Make plots of DataFrame using matplotlib / pylab - ‘barh’ : horizontal bar plot,‘hist’ : histogram
ax.legend(['TRUE','DECEPTIVE'])

'''
Observations in Positive Reviews:

Personal Pronoun (PRP) - I,he,she,it
The count of PRP(Personal pronoun) is higher is Deceptive compared to the true Reviews.

Possesive Pronoun (PRP$) - my, his, her
The count of PRP$(Possesive Pronoun) is higher is Deceptive compared to the true Reviews.

Verb, non-3rd person singular present (VBP)
am, are
The count of VBP(Verb, non-3rd person singular present) is higher is Deceptive compared to the true Reviews.

'''

        
#POS Tags Counts in Negative Reviews¶
fig, ax = plt.subplots(figsize=(20, 25))
pos_tag_count(df[df.polarity=='negative']).sort_values(by=['dec_cnt']).plot(kind='barh',rot=0,ax=ax)
#sort_values -> Sort by the values along either axis. by = Name or list of names to sort by
#plot - > Make plots of DataFrame using matplotlib / pylab - ‘barh’ : horizontal bar plot,‘hist’ : histogram
ax.legend(['NEGATIVE','DECEPTIVE'])

'''
Observations in Positive Reviews:

Personal Pronoun (PRP) - I,he,she,it
The count of PRP(Personal pronoun) is higher is Deceptive compared to the true Reviews.

Possesive Pronoun (PRP$) - my, his, her
The count of PRP$(Possesive Pronoun) is higher is Deceptive compared to the true Reviews.

Verb, non-3rd person singular present (VBP)
am, are
The count of VBP(Verb, non-3rd person singular present) is higher is Deceptive compared to the true Reviews.

'''
    
#POS Tags Combining Positive and Negative Sentiments

fig, ax = plt.subplots(figsize=(20, 25))
pos_tag_count(df).sort_values(by=['dec_cnt']).plot(kind='barh',rot=0,ax=ax)
ax.legend(['TRUE','DECEPTIVE'])

#Observations in Combined Positive and Negative Reviews:
#Personal Pronoun (PRP) - The count of PRP(Personal pronoun) is higher is Deceptive compared to the true Reviews.   

#Most Frequent Words in the True reviews

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

wordcloud = WordCloud(width=1600,height=1600,background_color='white',min_font_size=10).generate(".".join(df[df.label=='true'].reviews))
plt.figure(figsize=(12,12),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()



#Most Frequent Words in the Deceptive reviews


wordcloud = WordCloud(width=1600,height=1600,background_color='white',min_font_size=10).generate(".".join(df[df.label=='deceptive'].reviews))
plt.figure(figsize=(12,12),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

#Converting Text into Machine Interpretable Numbers

#Storing the text in X, that is out Indepnedent Variable and Storing the labels in y that is our dependent variable.

X = df.reviews
y = df.label

#Text Processing

def Processing(text):
    #Lower the text
    text = text.lower()
    
    #Passing the text into spacy Documnet object
    doc = nlp(text)
    
    # Extracting tokens out of SpaCy Document Object
    token = [str(t) for t in doc]
    
    # Removing punctuation
    
    token = [word.translate(str.maketrans('','',string.punctuation)) for word in token]
    
    #Remove word that contain number
    token = [word for word in token if not any(c.isdigit() for c in word)]
    
    #Remove empty token
    token = [t for t in token if len(t)>0]
    
    return token

'''
TFIDF Vectoriser
Using Sklearn TFIDFVectoriser
Everytime you run TFIDF vectoriser, you need to pass in a tokenizer otherwise it takes in TFIDF's 
default tokenizer.

You can see I passed my preprocessing function to the tokenier!! Whats happens there is when you call 
fit function on your trainning data, each datapoint in your review first passes to preprocess and then 
applies tfidf on list of words.

'''

from sklearn.feature_extraction.text import TfidfVectorizer    
Tfidf_vectorizer = TfidfVectorizer(tokenizer=Processing,ngram_range=(1,2),max_features =5000,lowercase=False)
# TfidfVectorizer -> Convert a collection of raw documents to a matrix of TF-IDF features
# tokenizer -> Override the string tokenization step while preserving the preprocessing and n-grams generation steps
# ngram_range -> The lower and upper boundary of the range of n-values for different n-grams to be extracted
# (1, 2) means unigrams and bigrams,
# max_features -> build a vocabulary that only consider the top max_features ordered by term frequency across the corpus

tfid =  Tfidf_vectorizer.fit_transform(X)   

dense = tfid.todense()
#Return a dense matrix representation of this matrix.
dense.shape    

#Converting dense to dataframe

x=pd.DataFrame(dense)

#Test Train split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
# train_test_split -> used for initializing the internal random number generator, which will decide the splitting of 
#data into train and test indices
#Setting random_state a fixed value will guarantee that same sequence of random numbers are generated each time 
#you run the code


#Model Building
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

from sklearn.metrics import accuracy_score
print('Traning set Accuracy - ',accuracy_score(y_train,lr.predict(X_train)))
print('Test set Accuracy - ',accuracy_score(y_test, y_pred))

# You can see that the model is overfitting. Our Trainning set accuracy is 96 % while our test set Accuracy is 89 %. 
#And that too with a very basic linear classifier. Try tunning the Hyperpameters:
# max_df,ngram_range,max_features

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer= Processing, lowercase=False)),
         ('clf', LogisticRegression())])


## Pass in the Parameters in the pipeline 
## as these parameters belong to the tfidf in pipeline thats why you need to specify tfidf__ before the parameters
    
parameters = {
    'tfidf__max_df': [0.25, 0.5, 0.75],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_features': [1000, 1500, 2000, 2500, 3000]
}    

grid_search_tune = GridSearchCV(pipeline, parameters, cv=5)
grid_search_tune.fit(X, y)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)

'''
We are training on 1200 data points with 5000 features, your model has been cursed by dimensionality. 
So even a linear classifier (Logistic Regression) is overfitting. The only way to reduce this overfitting is by 
getting more data which is not possible in our case otherwise do some feature selection to get best features 
our of these 5000 features.

In the Exploratory Data Anlysis Section we comapred the pos tags count in the True and Deceptive reviews. 
There I mentioned as a note we are going to use these in POS tag filtering. So, Instead passing all the text 
we are filtering th text for some POS tags which are going to help the classifier.

'''


#Filtering POS Tags
X[0]

from tqdm import tqdm

#tqdm -> derives from the Arabic word taqaddum (تقدّم) which can mean “progress,”
#Instantly make your loops show a smart progress meter - just wrap any iterable with tqdm(iterable), and you’re done!

#Allowed POS Tag
allowed_word_types = ['PRP','JJ','NNP','NNS','PRP$','VDB','VB']

#We need to create a vocab which will have all words with the above POS Tags

import nltk
nltk.pos_tag(['we'])

all_word = []

for p in tqdm(X):
    #Preprocessing the text
    words = Processing(p)
    #Getting the POS tag using NLTK pos_tag function
    pos = nltk.pos_tag(words)
    
    #############################################################
    # POS will have list of tupples where a tuple is a combination of word and POS Tag('we','PRP')
    ############################################################
    
    for w in pos:
        # w is a tupple and 2nd element in POS Tag
        if w[1] in allowed_word_types:
            #w[0] is the word
            all_word.append(str(w[0]))
pos_word_final = list(set(all_word))            
        
#Now TF-IDF vectorizer will form vocab on these words , that will form feature

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=Processing,ngram_range=(1,1),vocabulary=pos_word_final)
tfid = tfidf_vectorizer.fit_transform(X)
dense = tfid.todense()
x = pd.DataFrame(dense)

x.shape

#Feature Selection

from sklearn.feature_selection import SelectPercentile,chi2
X_new = SelectPercentile(chi2,percentile=10).fit_transform(x,y)

X_new.shape

#Test Train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_new,y,test_size=0.25,random_state=42)

#Model Building

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report,(y_pred,y_test))

from sklearn.metrics import accuracy_score
print('Training set score : ',accuracy_score(y_train,lr.predict(X_train)))
print('Test set score : ',accuracy_score(y_test,y_pred))

#There is serious overfit, this model might work well on the data it has seen but will fail on the unseen data.

#Using Pre-Trained word embedding

#Google Word2Vec

# first download Pretrained word vectors
import wget
wget.download('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')

# Load these vector using gensim

import gensim
google_word2vec=  gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
google_word2vec['hello'].shape

#when you pass in a word in google wordevec pretrained model, it return a 300 dimensional vector. 
#So, we are going to average out the word vectors for each row of data.

#The function below takes in a sentence and averages out the word vectors for the whole sentence, 
#so you get the 300 dimensional sentence as a result.

def get_mean_vector(words):
    #remove out-of vocabulary words
    words = [word for word in words if word in google_word2vec.vocab]
    if len(words)>1:
        return np.mean(google_word2vec[words],axis=0)
    else:
        return []
  
x = df.spacy.apply(lambda x : get_mean_vector([str(word) for word in x]))    
x.shape

#After appling mean vector, we get back 1600 dimension array, but it should be (1600, 300).
#Convert to DataFrame

x = pd.DataFrame(x)
x.head()

x= x.spacy.apply(pd.Series)
x.head()

#Test Train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

#Model Building

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))

#We have found our best vectorisation technique.
#We are going to go ahead with this word2vec pretrined Model and try to fit some other complex models



#Building Classification Models

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=42)

#XGBoost classifier

from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.02,n_estimator=400,objective='binary:logistic',silent=True,nthread=1)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

from sklearn.metrics import accuracy_score
print("Training Set Accuracy: ", accuracy_score(y_train, lr.predict(X_train)))
print("Test Set Accuracy: ", accuracy_score(y_test, y_pred))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (7,5))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g', xticklabels=['Deceptive', 'True'], yticklabels=['Deceptive', 'True'])
plt.xticks(rotation=45)
plt.yticks(rotation='horizontal')
plt.title("CONFUSION MATRIX", fontsize= 18)
plt.show();


#Formatting End Results


class deceptive:
    def __init__(self,text):
        self.text = nlp(text)
        
    def get_mean_vector(self,words):
        #Remove out of vocab words
        words = [word for word in words if word in google_word2vec.vocab]
        if(len(words)>=1):
            return np.mean(google_word2vec[words],axis=0)
        else:
            return []
        
    def predict(self):
        self.test = [str(t) for t in self.test]
        self.vector = self.get_mean_vector(self.test)
        return xgb.predict(pd.DataFrame([self.vector]))[0]
X[0]   
deceptive(X[0]).predict()













































    

















    













