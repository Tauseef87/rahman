#Wordnet is semantic based word dictionary some thing like theasarus
#Synonyms : beautiful = pretty , studying = reading

#Wordnet
from nltk.corpus import wordnet as wn
wn.synsets('beautiful')
#[Synset('beautiful.a.01'), Synset('beautiful.s.02')]
#Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet
#Synset is nothing but Sense for every word we are trying to find sense
#Here in output beautiful is coming in two sense adjective and adverb
wn.synsets('do')
wn.synset('beautiful.a.01').lemma_names()
#lemma_name is root word
wn.synset('car.n.01').lemma_names()

wn.synsets('stocking')
wn.synset('stocking.n.01').lemma_names()
wn.synset('stock.v.01').lemma_names()
