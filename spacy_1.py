import sys
sys.path.append("F:\Algorithmica\MyCodes")
import spacy
#F:\Algorithmica\MyCodes\NLPCodes
with open('F:\Algorithmica\MyCodes\Tripadvisor_hotelreviews.txt','r',encoding='utf-8') as d:
    reviews = d.read()
#encoding - it is nothing but ASCII version where different computer can be used to get on same page    
#load langauge library
nlp = spacy.load('en_core_web_sm')  #This language model,en_core_web_md(medium),en_core_web_lg(large) 
#Creating a doc object
doc = nlp(reviews)


#When to use nltk and spacy
'''Spacy is predominantly in speed. Some of the basic operations are very fast
NLTK on other hand is very robust. It's been built over years. So it can do some 
complex operations very well. Generally in production as speed is criteria - spacy will be preferred.
But if spacy isn't perfmorming well as per job - nltk only '''

''' When you pass a text to langauge model i.e nlp here, your text enters a processing pipeline that
first breaks down the text and perform a series of operation to tag,parse and desdcribe the data. so
we dont need to write code manually to tokenize,parse '''

''' In spacy we need to do tag,parsing or named entity manually as we used to do with nltk '''

#Pipeline
nlp.pipeline
#('tagger', <spacy.pipeline.pipes.Tagger at 0xeb84f54a8>) - tagging
#('parser', <spacy.pipeline.pipes.DependencyParser at 0xeb8607dc8>) - parsing
#('ner', <spacy.pipeline.pipes.EntityRecognizer at 0xeb8607e28>) - Named entity Recognisation
#(NER)is probably the first step towards information extraction that seeks to locate and classify 
#named entities in text into pre-defined categories such as the names of persons, organizations, 
#locations, expressions of times, quantities, monetary values, percentages, etc

#It is used for below cases :-
#Which companies were mentioned in the news article?
#Were specified products mentioned in complaints or reviews?
#Does the tweet contain the name of a person? Does the tweet contain this person’s location?

doc.text[:50]

#Sentences
# Sentences object is a generator object so we need list.
#A Python generator is a function which returns a generator iterator (just an object we can iterate over) by calling yield
for sen in list(doc.sents):
    print(sen)
list(doc.sents)[:5]

#Tokens
list(doc)[:10]

#list of token
len(doc)

doc1 = nlp("Tauseef isn't reading")
list(doc1)

#Stop words
print(nlp.Defaults.stop_words)
len(nlp.Defaults.stop_words)

nlp.vocab['Tauseef'].is_stop

nlp.Defaults.stop_words.add('Tauseef')
nlp.vocab['Tauseef'].is_stop = True


nlp.Defaults.stop_words.remove('Tauseef')
nlp.vocab['Tauseef'].is_stop = False

#POS- Tagging parts of speech Tagging
#To view the coarse POS tag use token.pos_
#To view the fine-grained tag use token.tag_
#To view the description of either type of tag use spacy.explain(tag)

print(doc[100].text,doc[3].pos_,doc[3].tag_,spacy.explain(doc[3].tag_))

for token in doc[:20]:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')

#Dependencies
for token in doc[:10] :
    print(token,'->',token.dep_)
    
#Count POS tagging
POS_counts = doc.count_by(spacy.attrs.POS)
#count_by - Count the frequencies of a given attribute
POS_counts

#Decode the attribute ID of count_by
doc.vocab[96].text

for K,v in sorted(POS_counts.items()) :
    print(f'{K}. {doc.vocab[K].text:{10}}:{v}')
    
#Count the different fine-grained tags
Tag_count = doc.count_by(spacy.attrs.TAG)    
for k,v in sorted(Tag_count.items()) :
    print(f'{k}. {doc.vocab[k].text:{10}}:{v}')
    
#visualize Parts of speech
from spacy import displacy
# Render the dependency parse of first sentence
displacy.render(list(doc.sents)[0], style='dep', jupyter=True, options={'distance': 110})




