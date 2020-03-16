import re
import nltk

#Using word from corpus

wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
wordlist[1:10]

# Search Function in regular Expression
# Ends with
list1 = [w for w in wordlist if re.search('ly$',w)][1:10]
print(list1)

#Starts with
list2 = [w for w in wordlist if re.search('^twi',w)]
print(list2)
 # twilight
list3 = [w for w in wordlist if re.search('^..il.g.t$',w)]
print(list3)

#find word with 3 place - j and 5 place - t
list4 = [w for w in wordlist if re.search('^..j..t..$',w)]
print(list4)
#Any two pair
list5 = [w for w in wordlist if re.search('^[gho][mno]$',w)]
print(list5)
# + indicates one or more - preceding character occurance
# * means zero or more - preceding character occurances
# - is for range [1-9]

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
list6 = [w for w in chat_words if re.search('^m+i+n+e+$',w)]
print(list6)

# [^words not starting with]
list7 = [w for w in chat_words if re.search('^[^aeiouAEIOU]+$',w)]
print(list7)

# \ operator means the following character i.e "." must be matched exactly
corp = sorted(set(nltk.corpus.treebank.words()))
list8 = [w for w in corp if re.search('^[0-9]+\.[0-9]+$',w)]
print(list8)

# exercise : Get the numbers which start with only 1 and have only 2 decimals after .
# {Exactly the specified number of occurrences} EG - {1 - atleast one occurance,2- maximam 2 occurane)}
list9 = [w for w in corp if re.search('^[1]+\.[0-9]{2}$',w)]
print(list9)

#First a to z atleast 5 occuance then - then a to z atleast 2 and max 5 and then ends a to z atleast 5 occurance
list10 = [w for w in corp if re.search('^[a-z]{5,}\-[a-z]{2,3}\-[a-z]{5,}$',w)]
print(list10)

#Pipe character

list11 = [w for w in corp if re.search('^[a-zA-Z]+(sed|zed)$',w)][1:50]
print(list11)


#Findall
#To find all patters inside a string and get the pattern rather than word

word = 'GradvalleyDataScience-NLPCourse-Week2'
re.findall(r'[aeiou]',word)

#FreqDist
#FreqDist = This is basically how many time each word occur
#FreqDist expects an iterable of tokens. A string is iterable --- the iterator yields every character
words = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for word in words for vs in re.findall(r'[aeiou]{2,}',word))
fd.items()

# Conditional Freq Distribution

words = nltk.corpus.toolbox.words('rotokas.dic')
pattern = [cv for w in words for cv in re.findall(r'[ptksvr][aeiou]',w)]
pattern = [w for w in words if re.findall(r'[ptksvr][aeiou]',w)]
cond_dict = nltk.ConditionalFreqDist(pattern)
cond_dict.tabulate()
# ConditionalFreqDist --> it shows the probability matrix of different word combination
# tabulate --> shows after k how many a is coming,after k how many e is coming or what is the 
# probability that ka will come to next ke or ki or ko
    
#print only the entire string instead of part of it
re.findall(r'^.*(?:ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')

#to get both parts () -> capture and group
re.findall(r'^(.*)(ing|ly|ed|ious|ies|ive|es|s|ment)$', 'processing')







