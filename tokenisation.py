raw = "When I'M a Duchess,' she said to herself, (not in a very hopeful tone... though), 'I won't have any pepper in my kitchen AT ALL. Soup does very... well without--Maybe it's always pepper that makes people hot-tempered,'.."
raw
print(raw.split())
#Here it split on spaces and doesnt consider any
import re
print(re.split(r' ',raw))

print(re.split(r'\s+',raw))
#instead of using a space , we use \s - any no white space character
#here we have difference re split and normal split

sample = 'I am typing (text data)'
print(re.split(r'\s+',sample))

re.split(r'\W+',sample)
re.split(r'\w+',sample)

# 'w' indicates split on any of [a-zA-Z0-9]
# 'W' indicates split on anything apart from [a-zA-Z0-9], if you came across to anything from [a-zA-Z0-9]
# It will split, only token it will take from [a-zA-Z0-9]
# r indicates raw string

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

# word_tokenizer is most sophistecated tokenizer of nltk
string1 = 'I am typing (text data)'
tokens1 = nltk.tokenize.word_tokenize(string1)
print(tokens1[:])

# The above code shows the difference between regaular expression and nltk token
# nltk word tokenizer much better than regular expression tokenizer
from nltk.corpus import RegexpTokenizer as regextoken
tokenizer = regextoken(r'\w+')

tokens = tokenizer.tokenize(string1)
print(tokens[:])

