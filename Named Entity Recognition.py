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

nlp.pipeline

# The named entities in the document. Returns a tuple of named entity Span objects
for ent in doc.ents[:50] :
    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    
'''
Doc.ents are token spans with their own set of annotations.

ent.text	The original entity text
ent.label	The entity type's hash value
ent.label_	The entity type's string description
ent.start	The token span's *start* index position in the Doc
ent.end	The token span's *stop* index position in the Doc
ent.start_char	The entity text's *start* index position in the Doc
ent.end_char	The entity text's *stop* index position in the Doc
'''

#Noun Chunks
#.text	The original noun chunk text.
#Doc.noun_chunks are base noun phrases
#Doc.root.text	The original text of the word connecting the noun chunk to the rest of the parse.
#.root.dep_	Dependency relation connecting the root to its head.

for chunk in doc[:50].noun_chunks:
    print(chunk.text+' - '+chunk.root.text+' - '+chunk.root.dep_+' - '+chunk.root.head.text)
    
#Visualising NERs
    
import warnings    
warnings.filterwarnings('ignore')

for sent in list(doc.sents)[:50]:
    displacy.render(nlp(sent.text), style='ent', jupyter=True)