# Stemming and lemmatisation
#stemming example
#Snowballs stemmer is the most popular stemmer
import nltk
nltk.SnowballStemmer(language = 'english').stem('automate')
token = ['waterloo','fortune','catchy','hired','trapping','inn','driven']
tokens_1 = [nltk.SnowballStemmer(language = 'english').stem(tokens) for tokens in token]
print(tokens_1)

porter = nltk.PorterStemmer()
for w in token:
    print(w," : ",porter.stem(w))

lancaster = nltk.LancasterStemmer()
for w in token:
    print(w," : ",lancaster.stem(w))

#which stemmer is to use depends on application. Porter stemmer is mostly used.
#Porter: Most commonly used stemmer without a doubt, also one of the most gentle stemmers
#SnowBall - Nearly universally regarded as an improvement over porter, and for good reason
#Lancaster: Very aggressive stemming algorithm, sometimes to a fault
    
#Lemmatisation
# It is cutting a word into root form it it exists in dictionary
lem = nltk.WordNetLemmatizer()    
# WordNet is nltk Lemmatizer
words = ['women','supreme','stocking']    
print([porter.stem(t) for t in words])
print([lem.lemmatize(t) for t in words])
#wherever a generator object is coming convert it into list
#stemming just chops the word
#Lemmatizer think ,checks in dic then chops

#Sentence tokenization
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
print(text[1:1000])
sents = sent_tokenizer.tokenize(text)
import pprint
pprint.pprint(sents[1:10])

#The pprint module provides a capability to “pretty-print” arbitrary 
#Python data structures in a well-formatted and more readable way

