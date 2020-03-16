import nltk
nltk.SnowballStemmer(language='english').stem('automote')

tokens = ['waterloo','fortune','catchy','hired','trapping','inn','driven']

tokens2 = [nltk.SnowballStemmer(language='english').stem(token) for token in tokens]
print(tokens2)

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

print([porter.stem(t) for t in tokens][1:10])

#Lemmatization

lem = nltk.WordNetLemmatizer()
words = ['women','supreme','stocking']
print ([porter.stem(t) for t in words])
print ([lem.lemmatize(t) for t in words])

#Here lemmatization does not look for verb and stem does

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')
print (text[1:1000])

sents = sent_tokenizer.tokenize(text)

import pprint

pprint.pprint(sents[1:10])

# Wordnet
from nltk.corpus import wordnet as wn
wn.synsets('beautiful')
wn.synsets('do')

wn.synsets('stocking')
wn.synset('stock.v.01').lemma_names()
wn.synsets('car')
wn.synset('car.n.01').lemma_names()

motorcar = wn.synset('car.n.01')
types_of_motorcar = motorcar.hyponyms()
types_of_motorcar

dog = wn.synset('dog.n.01')
dog.hypernyms()
dog.hypernym_paths()


#hyper means above and hypo means below in tree structure in word
cat = wn.synset('cat.n.01')
boar = wn.synset('boar.n.01')
dog = wn.synset('dog.n.01')
cat.lowest_common_hypernyms(boar)
cat.hypernym_paths()
boar.hypernym_paths()

cat.path_similarity(boar)
cat.path_similarity(dog)
