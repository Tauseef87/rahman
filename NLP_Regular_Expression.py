'''
[]	A set of characters [a-m]
\	Signals a special sequence (can also be used to escape special characters) '\d'
.	Any character (except newline character) 'he.o'
^	Starts with '^hello'
$	Ends with 'world$'
*	Zero or more occurrences 'ai*'
+	One or more occurrences 'ai+'
{}	Exactly the specified number of occurrences 'al{1}
|	Either or 'falls|fall'
()	Capture and group

Special Sequence

A special sequence is a \ followed by one of the characters
\A - Returns a match if the specified characters are at the beginning of the string "\AThe"
\b	- Returns a match where the specified characters are at the beginning or at the end of a word	r"\bain" r"ain\b"
\B	Returns a match where the specified characters are present, but NOT at the beginning (or at the end) of a word  r"\bain" r"ain\b"
\d	Returns a match where the string contains digits (numbers from 0-9)	"\d"
\D	Returns a match where the string DOES NOT contain digits	"\D"
\s	Returns a match where the string contains a white space character	"\s" and \S is opposite
\w	Returns a match where the string contains any word characters (characters from a to Z, digits from 0-9, and 
    the underscore _ character)	"\w"  and \W is opposite

Set

[arn]	Returns a match where one of the specified characters (a, r, or n) are present
[a-n]	Returns a match for any lower case character, alphabetically between a and n
[^arn]	Returns a match for any character EXCEPT a, r, and n
[0123]	Returns a match where any of the specified digits (0, 1, 2, or 3) are present
[0-9]	Returns a match for any digit between 0 and 9
[0-5][0-9]	Returns a match for any two-digit numbers from 00 and 59
[a-zA-Z]	Returns a match for any character alphabetically between a and z, lower case OR upper case
[+]	In sets, +, *, ., |, (), $,{} has no special meaning, so [+] means: return a match for any + character in the string
'''
#Findall Function - The findall() function returns a list containing all matches.
import re
import nltk
str = 'The rain in spain but in spain rain is not there'    
x = re.findall('spain',str)
print(x)

#search - It searches the string for a match, and returns match object
# if there is If there is more than one match, only the first occurrence of the match will be returned

p = re.search("\s",str)
print("First whiteSpace located at - ",p.start())

# Split function -  The split() function returns a list where the string has been split at each match
s = re.split("\s",str)
print(s)

s = re.split("rain",str,1)
print(s)

#number of occurrences can be controlled by specifying the maxsplit parameter:

# Sub -  sub() function replaces the matches with the text of your choice
 t = re.sub('rain','rein',str)
 print(t)

# Regex - supervisiedlearning.com
print(nltk.corpus.words.words('en'))
    
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower() ]

wordlist[1:10]
#twilight
list1 = [w for w in wordlist[1:100] if re.search('ly$',w)]
list2 = [w for w in wordlist if re.search('^twi',w)][1:10]
list3 = [w for w in wordlist if re.search('...li',w)][1:50]
list4 = [w for w in wordlist if re.search('^..il..ht$',w)]
list5 = [w for w in wordlist if re.search('^t.*t$',w)]
list6 = [w for w in wordlist if re.search('^[ghi][mno]$',w)]
list7 = [w for w in wordlist if re.search(w.isupper()|w.islower(),w)]
chatwords = sorted(set(w for w in nltk.corpus.nps_chat.words()))
list8 = [w for w in chatwords if re.search('^m+i+n+e+$',w)]
list9 = [w for w in chatwords if re.search('^y+e+s+$',w)]
list10 = [w for w in chatwords if re.search('^[^aeiouAEIOU]+$',w)]

corp = sorted(set(nltk.corpus.treebank.words()))
list11 = [w for w in corp if re.search('^[0-9]+\.[0-9]+$',w)][1:100]
corp1 = """ Hello How to find the complete sentences"""
list12 =[w for w in corp if re.search('^[1]+\.[0-9]+$',w)]
list12 = [w for w in corp if re.search('^[1]+\.[0-9][0-9]+$', w)]

#---------------------------------------------------------------------------------------------------------
list1 = [w for w in corp if re.search('^[1](\.[0-9]{1,2})$',w)]

list1 = [w for w in corp if re.search('^[0-9]{2}$',w)]
 # black-and-white
list1 = [w for w in corp if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}',w)]

list1 = [w for w in corp if re.search('^\w+(sed|zed)$',w)]


import urllib
url = 'http://stackoverflow.com/'
response = urllib.request.urlopen(url)
webcontent = response.read()
f = list(webcontent)
list1 = [w for w in f if re.search('^[html]$',w)]

#------------------------------------Findall-------------------------------------------------
#findall serach for any specific word eg D,a,t but not pattern 
word = 'DataScience-NLPCourse-Week2'
re.findall(r'[aeiou]',word)

words = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for word in words for vs in re.findall(r'[aeiou]{2,}', word))
fd.items()
print(type(fd))
# {a.b}- a is the lower ,b is upper limit 
#FreqDist = This is basically how many time each word occur
#Conditional Frequency distribution
words = nltk.corpus.toolbox.words('rotokas.dic')
patterns = [cv for w in words for cv in re.findall(r'[ptksvr][aeiou]',w)]
cond_dict = nltk.ConditionalFreqDist(patterns)
cond_dict.tabulate()
# tabulate --> shows after k how many a is coming,after k how many e is coming or what is the 
# probability that ka will come to next ke or ki or ko
# r' --> It means that escapes won’t be translated, r'\n' is a string with backslash followed by a letter
# ConditionalFreqDist --> it shows the probability matrix of different word combination
re.findall(r'^.*(ing|ly|ed|ious|ies|ive|es|s|ment)$','abbeded')
#need to ask for ?:
re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')

re.findall(r'^(.*)(ing|ly)$','processing')

###########################################################
from nltk.corpus import gutenberg,nps_chat
chat = nltk.Text(gutenberg.words())
chat.findall(r"<.*> <.*> <bro>") 
# here three word phrase .* first any character,.* any charactter ,third bro

p = nltk.corpus.indian.words('hindi.pos')
p[100]






