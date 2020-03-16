#Unigram Approach
from nltk.corpus import reuters

from collections import Counter
# Counter is used for frequency count of words,counters are accessed just like dictionaries
counts = Counter(reuters.words())
#total number of words
total_count = len(reuters.words())


#The most common 20 words
print(counts.most_common(n=20))
#most_common() is used to produce a sequence of the n most frequently encountered input values and their respective counts.

#Compute the frequency of words
for word in counts:
    counts[word] /= float(total_count)
# It is like Counts[word]=count[word]/ float(total_count)

import random
#Generate 100 words of language
text = []
for _ in range(100):
    r = random.random()
    accum = .0
    for word,freq in counts.items():
        accum += freq
       # print(accum)
        if accum >= r:
            text.append(word)
            break
print(' '.join(text))  

# In above we are generating 100 times the random value and taken it as frequency comparision with
# frequency of counts that has word with there frequncy. if the frequncy of word is > random frequncy value
# Then it is loaded in the list 

#Bigram Approach
#Eg The boy bought a toy -> <The boy>,<boy bought>,<bought a>,<a toy>
#p(The,boy)= count(The and boy)/count(The) etc

#One idea that can help us generate better text is to make sure the new word we’re adding to the 
#sequence goes well with the words already in the sequence.
#Let’s make sure the new word goes well after the last word in the sequence (bigram model) 
#or the last two words (trigram model)

from nltk import bigrams,trigrams
from collections import defaultdict

first_sentence = reuters.sents()
print(first_sentence)

#get the Bigrams
print(list(bigrams(first_sentence)))

#Get the padded bigrams
# Padded = As for first there is no bigram so it comes as None
print(list(bigrams(first_sentence,pad_left=True,pad_right=True)))

#The bigram model approximates the probability of a word given all the previous words
#defaultdict ->defaultdict means that if a key is not found in the dictionary, then instead of a KeyError 
#being thrown, a new entry is created. The type of this new entry is given by the argument of defaultdict
#Lambda is an expression and not a statement. It does not support a block of expressions.
#lambda argument : expression eg - add = lambda x,y:x+y,print(add(6,7))

bigram_model = defaultdict(lambda:defaultdict(lambda:0))
# So here we creating a nested dictionary so when you do bigram_model["ham"]["spam"], the key "ham" is inserted
# in y if it does not exist. The value associated with it becomes a defaultdict in which "spam" is 
#automatically inserted with a value of 0.

for sentence in reuters.sents():
    for w1,w2 in bigrams(sentence, pad_right=True, pad_left=True):
        print(w1,' ',w2)
        bigram_model[(w1)][w2] += 1
        #bigram_model[(w1)][w2] = bigram_model[(w1)][w2]+1
print(bigram_model["the"]["economists"])   
print(bigram_model["the"]["nonexistingword"]) 
print(bigram_model[None]["The"])
#"economists" follows "the" 8 times wherease nonexistingword follows "the" 0 times
#Transforming into probability

for w1 in bigram_model:
    print(bigram_model[w1].items())

for w1 in bigram_model:
    total_count = float(sum(bigram_model[w1].values()))
    for w2 in bigram_model[w1]:
        bigram_model[w1][w2] /= total_count
print(bigram_model["the"]["economists"])          
print(bigram_model[None]["The"])

#Trigram Approach
print(list(trigrams(first_sentence)))
#Padded
print(list(trigrams(first_sentence,pad_left = True,pad_right=True)))

trigram_model = defaultdict(lambda:defaultdict(lambda:0))

for sentence in reuters.sents():
    for w1,w2,w3 in trigrams(sentence,pad_left = True,pad_right = True):
        trigram_model[(w1,w2)][w3] += 1

print(trigram_model["what", "the"]["economists"])

# Economists follows What the 2 times

print(trigram_model["what", "the"]["nonexistingword"]) # 0 times
# nonexistingword follows What the 0 times

# N-Gram example using NLTK Language
from nltk.util import pad_sequence
#Returns a padded sequence of items before ngram extraction.
#pad_sequence(sequence,n,pad_left,pad_right,left_pad_symbol,right_pad_symbol)
#sequence: the source data to be padded,n: the degree of the ngrams,pad_left: whether the ngrams should be left-padded
#pad_right: whether the ngrams should be right-padded
#left_pad_symbol: the symbol to use for left padding (default is None)

from nltk.util import everygrams
#Returns all possible ngrams generated from a sequence of items, as an iterator
#sent = 'a b c'.split(),list(everygrams(sent,max_len=2))[('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c')]

from nltk.lm.preprocessing import pad_both_ends

#Pads both ends of a sentence to length specified by ngram order


from nltk.lm.preprocessing import flatten

from nltk.lm import MLE
#MLE - Maximum Likelyhood Estimator, Returns the MLE score for a word given a context.
model = MLE(3) #It is 3 gram model

train_data = [list(trigrams(sentence)) for sentence in reuters.sents()]

model.fit(text=train_data,vocabulary_text = reuters.words() )
#Here model.fit do nothing but it only calculate the probability as we did above

print(model.vocab)
model.counts["what", "the"]["economists"]
model.score("economists", ["what", "the"])

#We always represent and compute language model probabilities in log format as log probabilities.
#Since probabilities are (by definition) less than or equal to 1, the more probabilities we multiply together, 
#the smaller the product becomes
#Adding in log space is equivalent to multiplying in linear space, so we combine log probabilities by adding them.
#p1×p2×p3×p4=exp(logp1+logp2+logp3+logp4)


# Evaluating Language Models
# We need to find maximum probabilities outoff other different model probability
# If we have m1,m2,m3 model and we have p1,p2,p3 probability then we need to find max probability
# perplexity = 1/p1,1/p2,1/p3 we are trying to get maximum of probability and perplexity does it by minimizing the 1/p1
len(train_data)
test= train_data[:700]
model.perplexity(test)
