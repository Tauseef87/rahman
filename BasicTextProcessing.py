#Creating a string with Single quotes
my_string = 'Hello Everyone'
print(my_string)

#Creating a string with double quotes
my_string = "Hello Everyone"
print(my_string)

#Creating a string with Triple quotes
my_string = ''' Hello Everyone'''
print(my_string)

#Creating a string with Triple quotes - allow multiple lines
my_string = '''Hello Everyone,
you are welcome'''
print(my_string)

#single quotes or double quotes with \n - new line
my_string = 'Hello\nEveryone'
print(my_string)

my_string = "Hello\nEveryone"
print(my_string)

#Raw string to ignore escape sequence
my_string = r'Hello\nEveryone'
print(my_string) 

#------------------------------------------------------

#Accessing character in string
string = 'SupervisedLearning'
print('string = ',string)

string = 12344.789
print('string is float ',string)

#first charcater
print('String[0] = ',string[0])
#last character
print('String[-1] = ',string[-1])
#second last character
print('String[-2] = ',string[-2])
#2nd and 5th charcter
print('String[1:5] = ',string[1:5])
#6th to 2nd last charcter
print('String[6:-2] = ',string[5:-1])
#getting all the string characters skipping the every 2nd 
print('String[] = ',string[0::2])

#------------------------------------------------------------

#Concate
string1 = 'Hello Everyone'
string2 = 'Welcome to class'
print(string1+' '+string2)

#Multiple

String = 'Hello'
print('Hello 3 times = ',(String+' ')*3)

#Using join
print(" * ".join([string1,string2,'End']))

# In join left is the string to join and right inside parenthesis is the iterable

#----------------------------------------------------------

#String membership test

print('Hello' in string1)

print('Hello1' in string1)

#----------------------------------------------------------

# formating of string

# Default Order
string = "{}{}{}".format('Happy ','New ','Year')
print('Print in Default order')
print(string)

#Positional format
string = "{1}{0}{2}".format('Happy ','New ','Year')
print(string)

#kyword format
string = "{h}{n}{y}".format(h ='Happy ',n = 'New ',y = 'Year')
print(string)

# f-string

name = 'Tauseef'
age = 31
print(F"Hello , {name} you are {age} old")

# Time spped for f string
import timeit
string = """
name = "Tauseef"
age = 31
'{} is {}.'.format(name,age)
"""
timeit.timeit(string, number= 10000)

string= """
name = "Tauseef"
age = 31
f"{name} is {age}"
"""
timeit.timeit(string, number= 10000)

# Usage of f string
library = [('Author','Topic','Pages'),('Twain','Rafting',90),('Feyman','Physics',100)]

for book in library:
    print(f'{book[0]:{10}}{book[1]:{10}}{book[2]:{10}}')


# Commonly Used string functions
    
string = 'hello Everyone'

#first charcter capital

print(string.capitalize())

#All alphabetic character to lower case
string = 'HELLO ALL'
print(string.lower())

#first letter is converted to upper else other in lower

string = 'HELLO ALL'
print(string.title())

# Split
string = 'hello Everyone'
print(string.split())

string = 'hello-Everyone'
print(string.split('-'))

# with maxsplit =1 it split for one occurance
string = 'www.all.com'
print(string.split('.',maxsplit=1))

# PDF Reading
import PyPDF2
f = open('F:\Algorithmica\MyCodes\DAAI Newbie document.pdf','rb')
# rb - Read file in binary format
pdf_reader = PyPDF2.PdfFileReader(f)
#Number of pages
print(pdf_reader.numPages)

Page_one = pdf_reader.getPage(0)
print(Page_one)

# Extracting text from page1
page_one_text = Page_one.extractText()
page_one_text
f.close()



