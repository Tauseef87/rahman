import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns',100) #show all columns when looking at dataframe
import warnings
import matplotlib
from IPython.display import Math, HTML
import os
#change the working directory to aclimdb
os.chdir('F:/Imdb Dataset/aclImdb')
#Getting positive Train data
train_pos_list = list()
import glob
# Glob - It retrieves the list of files matching the specified pattern in the file_pattern parameter.
train_pos = '/Imdb Dataset/aclImdb/train/pos/*.txt'
pos_file = glob.glob(train_pos)
for name in pos_file:
    with open(name,encoding="utf8") as f:
        train_pos_list.append(f.read())
        
#Getting Negative Train data
train_neg_list = list()
train_pos = '/Imdb Dataset/aclImdb/train/neg/*.txt'
neg_file = glob.glob(train_pos)
for name in neg_file:
    with open(name,encoding="utf8") as f:
        train_neg_list.append(f.read())


#Creating a dataframe out of it
train_y = (['Positive'] * len(train_pos_list)) + (['Negative'] * len(train_neg_list))    
#Creating target variable as 2 class positive and Negative
train_x = train_pos_list
train_x.extend(train_neg_list)
#Now my train independent variable has both postive and negative comments
train = pd.DataFrame({'reviews':train_x,'Label':train_y})

#Getting positive test data
test_pos_list = list()
test_pos = '/Imdb Dataset/aclImdb/test/pos/*.txt'
pos_file = glob.glob(test_pos)
for name in pos_file:
    with open(name,encoding="utf8") as f:
        test_pos_list.append(f.read())
        
#Getting Negative test data  
test_neg_list = list()
test_neg = '/Imdb Dataset/aclImdb/test/neg/*.txt'
neg_file = glob.glob(test_neg)
for name in neg_file:
    with open(name,encoding="utf8") as f:
        test_neg_list.append(f.read())       
        
test_y = (['Positive'] * len(test_pos_list)) + (['Negative'] * len(test_neg_list))         
test_x = test_pos_list        
test_x.extend(test_neg_list)    
test = pd.DataFrame({'reviews':test_x,'Label':test_y})

train.head()
test.head()

train.shape
test.shape

font = {'family':'normal','size':15}
matplotlib.rc('font',**font)
#This change the size and width of label in plot
fig, (ax1, ax2) = plt.subplots(1, 2 ,figsize=(20,15))

ax1.pie(train.Label.value_counts().to_list(), labels=train.Label.value_counts().index.to_list(), autopct='%1.0f%%', pctdistance=.5, labeldistance=1.1) 
# pctdistance and labeldistance is to change the position of the labels and the values
ax1.legend( loc = 'upper right')
ax1.title.set_text("VALUE COUNTS OF OUR TARGET VARIABLE TRAINING SET")
ax2.pie(test.Label.value_counts().to_list(), labels=test.Label.value_counts().index.to_list(), autopct='%1.0f%%', pctdistance=.5, labeldistance=1.1) 
# pctdistance and labeldistance is to change the position of the labels and the values
ax2.legend( loc = 'upper right')
ax2.title.set_text("VALUE COUNTS OF OUR TARGET VARIABLE TESTING SET")
plt.show();

#Shuffle Data

from sklearn.utils import shuffle
train = shuffle(train)    
test = shuffle(test)        

train = train.reset_index(drop=True)        
test = test.reset_index(drop=True)        


#Basic Text Processing
train.isnull().sum()

#Detect and remove empty string
blank = []

for i,lb,rv in train.itertuples():  #Iterating over dataframe index,label,review
    if type(rv)==str:
        if rv.isspace():
            blank.append(i)

print(len(blank), 'Blanks :',blank)

#Sentence Length

def get_stat(column):
    pos = train.loc[train.Label =='Positive',column].describe()
    neg = train.loc[train.Label =='Negative',column].describe()
    data = pd.DataFrame({"Positive": list(pos), "Negative": list(neg)}, index=train[column].describe().index)
    return data

train['Word_Count']= [len(review.split()) for review in train['reviews']]

get_stat('Word_Count')
# It shows that how may postive comments are there and how many negatives are there

sns.set()
#Set is an unordered collection that does'nt allow duplicate.
data1 = train.loc[train.Label=='Positive','Word_Count']
data2 = train.loc[train.Label=='Negative','Word_Count']
fig = plt.figure(figsize=(19, 8))
sns.distplot(data1, hist = False, kde = True, 
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 label= 'Positive')
sns.distplot(data2, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}, 
                 label= 'Negative')
plt.title("Compare Word Length of Positive and Negative");     

#There is not much difference in word length but the max value for some Positive Reviews are higher
#because, may be some people got very excited after the twisting Climax of Movie

#Capitalization

train['upper_case']=[sum(char.isupper() for char in review) for review in train['reviews']]

#Trying to find how many upper case characters are there every review

get_stat('upper_case')
'''
     Positive      Negative
count  12500.000000  12500.000000
mean      37.370320     34.880800
std       36.479239     32.471106
min        0.000000      0.000000
25%       16.000000     16.000000
50%       26.000000     25.000000
75%       46.000000     43.000000
max      673.000000    749.000000
'''
#Not much difference between positive and negative reviews

#Usage of punctuation

import string
train['punctuation'] = [sum(char in string.punctuation for char in review) for review in train['reviews']] 
get_stat('punctuation')


'''
 Positive      Negative
count  12500.000000  12500.000000
mean      52.187040     54.012880
std       44.222677     44.331114
min        0.000000      1.000000
25%       23.000000     25.000000
50%       38.000000     41.000000
75%       68.000000     68.000000
max      543.000000    475.000000
'''

#IQR is same in both, but max value is higher for positive reviews.

#Dig in more and check why there is more number a exclamation mark in a single comment.

from collections import Counter
from nltk.corpus import stopwords

def getMostCommonWords(reviews,n_most_common,stopwords=None):
    #Flatten review column into a list of words and set each to lowerCase
    flattend_review = [word for review in reviews for word in review.lower().split()]
    
    #remove punctuation
    flattend_review = [''.join(char for char in review if char not in string.punctuation) for review in flattend_review]
    
    #remove stopwords
    if stopwords:
        flattend_review = [word for word in flattend_review if word not in stopwords ]
    
    #Remove any empty string that were created by the above process
    flattend_review = [review for review in flattend_review if review ]  
    return Counter(flattend_review).most_common(n_most_common)


positive_reviews = train.loc[train.Label == 'Positive', 'reviews']

negative_reviews = train.loc[train.Label == 'Negative', 'reviews']    

# Top twenty words in positive review with Stopwords
getMostCommonWords(positive_reviews,20)    

# Top twenty words in positive review without Stopwords
getMostCommonWords(positive_reviews,20,stopwords.words('english'))    

#Top twenty words in Negative review with Stopwords
getMostCommonWords(negative_reviews,20)  

# Top twenty words in Negative review without Stopwords
getMostCommonWords(negative_reviews,20,stopwords.words('english'))   

'''
Right away, we can spot a few differences, such as the heavy use of terms like "good", "great" in the positive class
, and words like "dont" and "bad" in place in the negative class. Additionally, if you increase the value of 
the n_most_common parameter in our function, you can see words like "not" (which nltk's corpus classifies as a stopword).
'''

import string
import re
from nltk.tokenize import word_tokenize
TAG_RE = re.compile(r'<[^>]+>')
#Here in re pattern - starts with < then [a set of starts with anything and end at > + >]
#Compile is used - We can combine a regular expression pattern into pattern objects, which can be used for pattern matching
def clean_text(text):
    #Lower text
    text = text.lower()
    
    #Remove HTML tags like <br>
    text = TAG_RE.sub('',text)
    
    #The re.sub() function in the re module can be used to replace substrings
    
    #Remove punctuation
    text = text.translate(str.maketrans('','',string.punctuation))
    #Maketrans will return a table for translate method to delete or change the string from the table returned from maketrans
    #Here all punction will be returned from maketrans that is returned  as dictionary with Key = value of punction and value as none
    #Transalte will return as string that is changed or deleted from table of maketrans
    # https://www.youtube.com/watch?v=ZCoHZNg3RPk - Take help from youtube
    # str.maketrans('','',string.punctuation) this will return asiic value of punctuation as key and None as value
    #Tokenize text
    text = [word for word in word_tokenize(text)]
    #Remove word that contains number
    text = [word for word in text if not any(c.isdigit() for c in word)]
    #Remove empty token
    text = [t for t in text if len(t)>0]
    #Remove word with one letter
    text = [t for t in text if len(t)>1]
    #join all
    text = " ".join(text)
    return text
   
train['review_clean'] = train['reviews'].apply(lambda X : clean_text(X))    
test['review_clean'] = test['reviews'].apply(lambda X : clean_text(X))    

#Lets build Word cloud on train and test data

from wordcloud import WordCloud  
import matplotlib.pyplot as plt
import pandas as pd

wordcloud = WordCloud(width = 1000, height = 1000, 
                background_color ='white', 
                min_font_size = 10).generate(str(train.loc[train.Label == 'Positive', 'review_clean'])) 
# plot the WordCloud image                        
plt.figure(figsize = (25, 20), facecolor = None) 
plt.imshow(wordcloud) 
#imshow - Display an image, i.e. data on a 2D regular raster.
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Word Cloud For Positive reviews in Training Dataset", fontdict={'fontsize': 25})
plt.show() 


# WordClou for Negative words 
wordcloud = WordCloud(width = 1000, height = 1000, 
                background_color ='white', 
                min_font_size = 10).generate(str(train.loc[train.Label == 'Negative', 'review_clean'])) 
# plot the WordCloud image                        
plt.figure(figsize = (25, 20), facecolor = None) 
plt.imshow(wordcloud) 
#imshow - Display an image, i.e. data on a 2D regular raster.
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Word Cloud For Positive reviews in Training Dataset", fontdict={'fontsize': 25})
plt.show() 

   
#Vectorization

from gensim.models import Word2Vec
Bigger_list = list()

for i in train['review_clean']:
    li = list(i.split(" "))
    Bigger_list.append(li)
    
Model = Word2Vec(Bigger_list,min_count=15,size=5000,sg=0)
# min_count -> Words below the min_count frequency are dropped before training occurs
# size = Dimensionality of the word vectors.
#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.

#Creating a CBOW Model

#function to avarage all word vector in a paragraph

def featureVecMethod(words,model,num_features):
    #Pre-initialising empty numpy array for speed
    featurevec= np.zeros(num_features,dtype='float32')
    nwords =0
    index2word_set = set(model.wv.index2word)
    
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featurevec= np.add(featurevec,model[word])
    
    # Dividing the result by number of words to get average
    featurevec = np.divide(featurevec, nwords)
    return featurevec        


# Function for calculating the average feature vector
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, 5000)
        counter = counter+1
        
    return reviewFeatureVecs

# Calculating average feature vector for training set
clean_train_reviews = []
for review in train['review_clean']:
    clean_train_reviews.append(review.split())
    
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, Model, 5000)

# Calculating average feature vactors for test set     
clean_test_reviews = []
for review in test['review_clean']:
    clean_test_reviews.append(review.split())
    
testDataVecs = getAvgFeatureVecs(clean_test_reviews, Model, 5000)

X_train= trainDataVecs
y_train= train.Label.map(lambda x: 1 if x=='Positive' else 0)
X_test= testDataVecs
y_test= test.Label.map(lambda x: 1 if x=='Positive' else 0)

#Classifing Positive and Negative Sentiment

#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#Predicting Test set
y_pred = classifier.predict(X_test)

# Evaluating result

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
cm = confusion_matrix(y_test,y_pred)
acc= accuracy_score(y_test, y_pred)
precision= precision_score(y_test, y_pred) # tp / (tp + fp)
recall= recall_score(y_test, y_pred) # tp / (tp + fn)
f1= f1_score(y_test, y_pred)

# Printing the Confusion Matrix with test Accuracy
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (7,5))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xticks(rotation=45)
plt.yticks(rotation='horizontal')
plt.title("CONFUSION MATRIX", fontsize= 18)
plt.show();
        
# Using Sentiword

#One of the most straightforward approaches is to use SentiWordnet to compute the polarity of the 
#words and average that value
#The plan is to use this model as a baseline for future approaches. 
#It’s also a good idea to know about SentiWordnet and how to use it


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize,word_tokenize,pos_tag

lemmatizer = WordNetLemmatizer()

def penn_to_wn(tag):
    '''
    Convert between PennTreeBank tag to simple wordnet Tags
    '''
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def swn_polarity(text):
    sentiment =0.0
    token_count = 0
    
    text = clean_text(text)
    raw_sentence = sent_tokenize(text)
    
    for raw_sentences in raw_sentence:
        tagged_sentence = pos_tag(word_tokenize(raw_sentences))
        
    for word,tag in tagged_sentence:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue
        lemma = lemmatizer.lemmatize(word,pos=wn_tag)
        if not lemma:
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
                continue
        # Take the first sense , the most commeon
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        token_count += 1
    # judgment call ? Default to positive or negative
    if not token_count:
        return 1
 
    # sum greater than 0 => positive sentiment
    if sentiment >= 0:
        return 1
 
    # negative sentiment
    return 0    
        
test.columns

print(swn_polarity(test.reviews[0]), test.Label[0])
print(swn_polarity(test.reviews[0]), test.Label[1])
print(swn_polarity(test.reviews[0]), test.Label[2])
print(swn_polarity(test.reviews[0]), test.Label[3])


