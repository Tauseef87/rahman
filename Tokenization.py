raw = "When I'M a Duchess,' she said to herself, (not in a very hopeful tone... though), 'I won't have any pepper in my kitchen AT ALL. Soup does very... well without--Maybe it's always pepper that makes people hot-tempered,'.."
raw

print(raw.split())
import re

print(re.split(r' ',raw))

print(re.split(r'\s',raw))

sample = 'I am typing (text data)'
sample2 = 'I am typing text data'
re.split(r'\W+',sample2)
re.split(r'\w+',sample)
re.split(r'\s+',sample)

import nltk
string = '''
At Waterloo we were fortunate in catching a train for Leatherhead, where we hired a trap at the station inn and drove for four or five miles through the lovely Surrey lanes. 
It was a perfect day, with a bright sun and a few fleecy clouds in the heavens. 
The trees and wayside hedges were just throwing out their first green shoots, and the air was full of the pleasant smell of the moist earth. To me at least there was a strange contrast between the sweet promise of the spring and this sinister quest upon which we were engaged. 
My companion sat in the front of the trap, his arms folded, his hat pulled down over his eyes, and his chin sunk upon his breast, buried in the deepest thought. 
Suddenly, however, he started, tapped me on the shoulder, and pointed over the meadows.
'''
tokens = nltk.tokenize.word_tokenize(string)
print(tokens[:40])

from nltk.corpus import RegexpTokenizer as regextoken
tokenizer = regextoken(r'\w+')

tokens = tokenizer.tokenize(string)
print(tokens[:40])

tokens = [token.lower() for token in tokens]
print(tokens[:20])


#\w Returns a match where the string contains any word characters (characters from a to Z, digits from 0-9, 
#and the underscore _ character) 	"\w" 	
#\W 	Returns a match where the string DOES NOT contain any word characters 	"\W"
