import re
re.search('n','\n') #first item is pattern and second item is string
#when we run above code , we did'nt get anything because in python '\n' means new line, it is not
#two character , it is a single character i.e new line

re.search('n','\\n')

#But when we give double \\ we will get the result, so a extra back slash treats it as 2 character

#But if we have more than one \n
re.search('n','\n\n\n\n') #Giving double back slash is not a good idea

#The best way to handle this is to convert it into raw string by using r'

re.search('n',r'\n\n\n')

#Regular has there own special character
#'\n' or r'\n both have a meaning of newline in regex

re.search('\n','\n\n\n') # Here in first parameter it is regex pattern while 2nd parameter is string
#The first parameter is a new line a/c to regex

re.search(r'\n','\n\n\n' )

#We get search result in both the case
#But when we convert the string to raw string then we will get no result
re.search(r'\n',r'\n\n\n' )

#So converting a string with raw string in 2nd parameter will affect the meaning of backslash
#But if we add r' to first parameter then it will change to regex own meaning i.e r'\n is new line in regex

####################################################################################################

# Methods of regex Match and Search

# re.search(pattern,string,flag)
# Pattern - The pattern that needs to search from string
# String - The string which will used to find pattern
# flag- Special option , it helps to situation where we neeed to find multiline
# The difference between match and search is 
# Match - It searches for a pattern only at begining of the string
# Search - It searches for a pattern every where of the string

re.match('c','abcdef') #No result found because c is not present at beginging of string
re.search('c','abcdef') #Ressult found because c is present in the string, span(2,3) means the search value
#found at 2nd location and ends at 3rd location


#We can use boolean value also
bool(re.match('c','abcdef'))
# It is only a way how none value can be converted to false

#The problem with search is it only checks for 1 occurance from whole string
re.search('c','abcdcf') # Here span(2,3) shows that 2nd location and ends at 3rd location 
#wherease the c also present at 4 position
# Search also works for multi line or new line
re.search('c','abdef\nc') # here we get the match at span(6,7)

re.match('c','abdef\nc') # Match does'nt give any result with newline

#To take print of match pattern
re.match('a','abcdef').group() # here in paraenthesis of group default value is 0

#if we want start and end value of span 
re.search('c','abcdef').start()
re.search('c','abcdef').end()

#Literal matching
re.search('na','abcdefnc abcd') #We didnt get any result as 'na' is a single pattern to be found in string

#But when we write 
re.search('n|a','abcdefnc abcd') #here it is searching for either n or a
#here a returned because it searches for first instance of either n or a , which may come first
#And here a comes first
# we may write more than or condition
re.search('n|a|b','abcdef nbca')

# Findall

#Search will pull out only first instance where findall will search only instance
re.findall('n|a','bcdefnc abcda') # It will return a list with all presence of n or a in string

#Multiple character

re.search('abcd','abcdef abcd') #It will find occurance of abcd at span(0,4)
re.findall('abcd','abcdef abcd') #It will give list of abcd at two occurance

###########################################################################################

#Character Set

#It can match a set of character

#\w - matches alpha numeric charcater - ie. a-z,A-Z,0-9, It represents any character in the set [a-zA-Z0-9_]
re.search(r'\w\w\w\w','abcdefnc abcd')
# Here 4 alpha numeric charcter is searched , it can be anything i.e abcd or 12m
re.search(r'\w\w\w\w','ab_defnc abcd') # here we get a12d as it is alpha numeric character

re.search(r'\w\w\w\w','ab_.efnc abcd') #Here we did'nt find any symbol . in [a-zA-Z0-9_] So efnc came as result

#Now \W - upper case W any thing which is not included in lower case w is taken by upper case W
re.search(r'\w\w\W','ab.efnc abcd') # So here lower case w does'nt include . and upper case W does, so in
#result we find .
re.search(r'\w\w\W\W','ab. fnc abcd') #here empty set is also printed 

#Quatifier - It is also a quanity measurement
#Quatifier comes after character
# + = 1 or more - gridy quantifier
# ? = 0 or 1
# * = 0 or more
# {n,m} = n to m repetations {,3},{3,} n = least and m = most amount or lower bound to upper bound

re.search(r'\w\w','abcdef abcd') # So we find 2 charcter ab
re.search(r'\w+','abcdef abcd') # So here 1 or more character came abcdef but not blank space
re.search(r'\w?','abcdef abcd') # So here 1 or one character came a
re.search(r'\w*','abcdef abcd') # So here 0 or more character came abcdef

re.search(r'\w+\W+\w+','abcdef abcd') # so here all alpha numeric character and blank space and all alpha numeric

#Pulling out specific amount
re.search(r'\w{3}','aaaaaaaaaaa') #Only 3 alpha numeric character
re.search(r'\w{1,3}','aa.') #Start at 1 and till 3 but here since at 3rd position we have . so result is aa

re.search(r'\w{1,10}\W{0,10}\w+','abcdef abcd') # 1-10 alpha numeric character of lower w
                                                # 0-10 upper W character
                                                # 1 or more alpha numeric character of lower w
                                                

######################################################################################################
                                                
#Other type of character set

#'\d - matches digits [0-9]
#'\D - matches any non digit charcter - ~\d any thing that \d does'nt work for

string = '23abcde++'                                                
re.search('\d+',string).group() #Here 1 or more digit printed 23

#'\s' - matches any whitespace character i.e newline,tab,spaces
# '\S' - matches any non whitespace character - ~\s

re.search('\S+',string).group() #No whitespace , so full string is grabed

string = '''Robots are branching out. A new prototype soft robot takes inspiration from plants by growing to explore its environment.

Vines and some fungi extend from their tips to explore their surroundings. 
Elliot Hawkes of the University of California in Santa Barbara 
and his colleagues designed a bot that works 
on similar principles. Its mechanical body 
sits inside a plastic tube reel that extends 
through pressurized inflation, a method that some 
invertebrates like peanut worms (Sipunculus nudus)
also use to extend their appendages. The plastic 
tubing has two compartments, and inflating one 
side or the other changes the extension direction. 
A camera sensor at the tip alerts the bot when it’s 
about to run into something.

In the lab, Hawkes and his colleagues 
programmed the robot to form 3-D structures such 
as a radio antenna, turn off a valve, navigate a maze, 
swim through glue, act as a fire extinguisher, squeeze 
through tight gaps, shimmy through fly paper and slither 
across a bed of nails. The soft bot can extend up to 
72 meters, and unlike plants, it can grow at a speed of 
10 meters per second, the team reports July 19 in Science Robotics. 
The design could serve as a model for building robots 
that can traverse constrained environments

This isn’t the first robot to take 
inspiration from plants. One plantlike 
predecessor was a robot modeled on roots.'''

(re.findall('\S+',string)) #It is getting all the words that does'nt have any space
' '.join(re.findall('\S+',string)) # It removes all the spaces and join words and return the article

# . - The dot matches any character except new line


string = '''Robots are branching out. A new prototype soft robot takes inspiration from plants by growing to explore its environment.

Vines and some fungi extend from their tips to explore their surroundings. Elliot Hawkes of the University of California in Santa Barbara and his colleagues designed a bot that works on similar principles. Its mechanical body sits inside a plastic tube reel that extends through pressurized inflation, a method that some invertebrates like peanut worms (Sipunculus nudus) also use to extend their appendages. The plastic tubing has two compartments, and inflating one side or the other changes the extension direction. A camera sensor at the tip alerts the bot when it’s about to run into something.

In the lab, Hawkes and his colleagues programmed the robot to form 3-D structures such as a radio antenna, turn off a valve, navigate a maze, swim through glue, act as a fire extinguisher, squeeze through tight gaps, shimmy through fly paper and slither across a bed of nails. The soft bot can extend up to 72 meters, and unlike plants, it can grow at a speed of 10 meters per second, the team reports July 19 in Science Robotics. The design could serve as a model for building robots that can traverse constrained environments

This isn’t the first robot to take inspiration from plants. One plantlike predecessor was a robot modeled on roots.'''

re.search('.+',string).group()
#'Robots are branching out. A new prototype soft robot takes inspiration from plants by growing to explore its environment.'
# The above line is returned because dot will all character except new line

re.search('.+',string,flags = re.DOTALL).group() # It will include the new also

#Creating Your own character set
#[A-Z] - It means A to Z , '-' is a metacharacter   It include all upper letter

string = 'Hello , There , How , Are , You'
re.findall('[A-Z]',string) # It will pull all upper case charcter
re.findall('[A-Z, ]',string) # It will pull all upper case charcter and a comma, here , is a charactter that we need to search

string = 'Hello , There , How , Are , You...'
re.findall('[A-Z,.]',string) # Same here dot is working as a charcter not like re dot i.e '.+' in line 192

re.findall('[A-Za-z,\s.]',string) # Here we are pulling A-Z capital,a-z small,comma,any space or whitespace and Dot

####################################################################################################

#Quatifier with Custom set

string = 'HELLO, There, How, Are, You...'
re.search('[A-Z]+',string) # Here HELLO is printed in upper case as we are asking for 1 or more capital case

re.findall('[A-Z]+',string) # Here 'HELLO', 'T', 'H', 'A', 'Y' is found as findall searches for whole string

re.findall('[A-Z]{2,}',string) # Here 'HELLO' 2 or more findall searches for whole string

re.search('[A-Za-z\s,]+',string) # one or more of 4 types of character

re.findall('[A-Z]?[a-z\s,]+',string) # Here ? means 0 or 1 and lower a-z \s and , for 1 or more

re.search('[^A-Za-z\s,]+',string) # Carrot inside of custom bracket means not this. Means not like A-Za-z\s

re.findall('[^A-Za-z]',string) # Not A-Z or Not a-z

#Groups
#Groups allow us to pull out section of a match and store them
string = 'John has 6 cats but I think my friend Susan has 3 dogs and Mike has 8 fishes'

re.findall('[A-Za-z]+ \w+ \d+ \w+',string) # So here we are trying to find Any upper or lower case 1 or more followed by 
                                           # space and character followed by space and number followed by character

re.findall('([A-Za-z]+) \w+ \d+ \w+',string) # So here if you see we have given () inside 'qutoes' this is group
                                             # Actually it is trying to make a group of A-Z and a-z 
                                             # here it gives John Susan Mike because when it started at beginging 
                                             # John is first group came then there space then Susan ..
                                           
                                         
match = re.search('([A-Za-z]+) \w+ (\d+) (\w+)',string) # Here ([A-Za-z]+) = Group1,(\d+)=Group2,(\w+)=Group3
match.groups()
match.group(1)
match.group(1,2)

#Span - start and end
match.span() # It is showing the begining and end of match string - john is 'j' = 0 and end at 15
match.span(0) # Group 0 start and end location

# find all has no group function
re.findall('([A-Za-z]+) \w+ (\d+) (\w+)',string).group(1) # It will throwh the error
re.findall('([A-Za-z]+) \w+ (\d+) (\w+)',string) # It will return as list
re.findall('([A-Za-z]+) \w+ (\d+) (\w+)',string)[0] # It will return a tupple as slicing

data = re.findall('(([A-Za-z]+) \w+ (\d+) (\w+))',string) #There we are putting small groups into large group
                                                          # 'John has 6 cats' This is main group
                                                          # 'John', '6', 'cats' - This is sub group 

for i in data:
    print(i[0])
    
# We can use Iteration

it = re.finditer('(([A-Za-z]+) \w+ (\d+) (\w+))',string) #Finditer -> It takes complete set of data and then 1 by 1 data 
                                                         # is pulled from iter variable 
next(it).group()    

for element in it:
    print(element.group())
    
#####################################################################################################

#Quantifier with Group

string = 'New York, New York 11369'    

#([A-Za-z\s]+) -> city group
#([A-Za-z\s]+) -> State group
#(\d+) -> Number Group

match = re.search('([A-Za-z\s]+),([A-Za-z\s]+) (\d+)',string)
match
match.group(1),match.group(2),match.group(3),match.group(0) # Here we are representing the group with number
                                                            # i.e group[1]=city,group[2]=state and ... so it might be tough
                                                            # to remember the number if group is big so we take name in group
#To name a group - ?P<group name> , group name inside the <>,followed by RE for group
#(?P<city>)


pattern = re.compile('(?P<City>[A-Za-z\\s]+),(?P<State>[A-Za-z\\s]+)(?P<ZipCode>\\d+)')
#Here we are just saving the pattern of regular expression in a variable
match = re.search(pattern,string)     
match.group('City'),match.group('State'),match.group('ZipCode')                                                      
                                                            
#######################################################################################################

#Split

#Example 1

re.split('\.','Today is Sunny. I want to go the park. I want to ride by-cycle')

#Include split point
re.split('(\.)','Today is Sunny. I want to go the park. I want to ride by-cycle')

#split with point another example
split = '.'
[i+split for i in re.split('\.','Today is Sunny. I want to go the park. I want to ride by-cycle')]

#Try to split at each tag

string = '<p>My mother has <span style="color:blue">blue</span> eyes. </p>'
#so we write for alpha numeric character
re.split('<\w+>',string) # It will work as spaces and quatation mark is not port of alpha numeric

re.split('<.+>',string) # here we are getting 2 empty string becaz at first it looks for <p> and last </p>, + is greedy quantifier

re.split('<[^<>]+>',string) # when ^ inside [] means negates i.e any charcter that does'nt have <>
                            # so it will not take <> from entire string
                            
#Handling empty string

[i for i in re.split('<[^<>]+>',string)]                            

#Alternative method

re.findall('>([^<]+)<',string) # here it starts with open > and take pattern which has does'nt start with < and then has
                               # end up with <
                               
#Another example

string = ',happy , birthday,'
list(filter(None,string.split(',')))    # here none any element that has no meaning i.e space,we filter it out
                                        # filter is generator so list is used

#re.sub - It utilizes regular expression and then substitute series of words from output of regular expression
string ="""U.S. stock-index futures pointed
to a solidly higher open on Monday, 
indicating that major 
benchmarks were poised to USA reboundfrom last week’s sharp decline, 
\nwhich represented their biggest weekly drops in months."""

print(re.sub('U.S|US|USA','United States',string)) # Here U.S|US|USA= Regular Expression,United States=Substitute word
                                                   # string = Original String. This is what re.sub works
                                                   
#Using function with sub

#Brief explanation with lambda
def square(x):
    return (x**2)

#With lambda

square = lambda x : x**2 # Here x before : is input and x**2 after : is output
                         # Lambda is quick way of creation of function which is small
square(3)

string = 'Dan has 3 snails. Mike has 4 cats. Alisa has 9 monkey'
#we are going to square digits i.e 3 to 9,4 to 16 and 9 to 81

re.search('(\d+)',string).group()
#It gives 3 as search will find first occurance
re.findall('(\d+)',string)

re.sub('(\d+)','1',string) # Here all digit is substituted with 1.

re.sub('(\d+)',lambda x:str(square(int(x.group(0)))),string)

# step 1 lambda x : x.group x is matching object and x is output of (\d+) this regular expression
# group(0) represents all group i.e not a specific group 1 or 2
# Turn the result into int
# use square function
# turn back to string

#Another example
input = 'eat laugh sleep study'
result = re.sub('(\w+)',lambda x:x.group(0)+'ing',input)
print(result)

#Backrefrencing

string = 'Merry Merry Christmas' 
#We try to remove duplicate by backrefrencing
re.sub(r'(\w+) (\1)',r'Happy \1',string)
#So here we are putting group 1 with 'Happy' that was previously Merry

##############################################################################################################

#Word Boundaries

'''
\b - is called 'boundary' and allows to isolate word

- is similar to ^ and $ , it checks for location

'''

string = 'Cat Catherine Catholic wildCat copyCat unCatchable'

pattern = re.compile('Cat')

re.findall(pattern,string)
#It just to see for Cat from all the string But we want only Cat word not Cat from Catherine 

#We use boundaries

pattern = re.compile(r'\bCat\b') #We are making a boundary from left and right to cat
re.findall(pattern,string)

#How \b works - It looks for both sides , left and right of boundary(\b) and insure that one side has one alphanumeric chr.
# and other side does'nt have alphanumric chr. so for word Catherine ,(Cat \b herine) it checks for chr left of boundary 
# i.e \b  and find alpha numeric chr and right to \b find another alpha numeric chr - so it does'nt satisfy condition
# But for 'wildCat' wild\bcat it checks for left and find alpha numeric chr i.e 'd' but in right non-alpha numeric chr(space) 
# And it satify the condition
###########################################################################################################

#Capture
#Consume

#Capture is mainly used in connection with groups . When group output something then it is known as capture group
#So Group has 2 forms capture and Non Capture Group , Non Capture means the group matches with pattern but does'nt
#return anything and if it returns then it is capture group

#Consume - for example if we have 'Welcome to python'
# and we have pattern ('\w+') so for finding the pattern the cursor will start from W of Welcome and till e
# so regular expression consume From W to e and will consume Welcome. But when it goes after e then Space comes
# which is nothing but Non-Alphanumric character so cursor will to next word to and check for consumption

#Look Around dont consume, They allow us to conirm some sort of subpattern is ahead or
# behind main pattern

# 4 Types of Look Around
# 1. ?= ->Positive look around
# 2. ?! -> Negative look around(It means that the group does'nt have any subpattern ahead  main pattern)
# 3. ?<= -> Positive look behind
# 4. ?<! -> Negative look behind (It means that the group does'nt have any subpattern behind  main pattern)


#Similar Syntax
# ?: Non-capture group
# ?p Naming group


#Example

'''
In the below string we are looking to consume the second column value",
only if the first column starts with ABC and the last column ",
has the value 'active'
So only the first row and last row satisfies this condition which will
output the value '1.1.1.1' and 'x.x.x.x
    
'''
string =     '''   
              ABC1    1.1.1.1    20151118    active
              ABC2    2.2.2.2    20151118    inactive
              ABC3    x.x.x.x    xxxxxxxx    active
              '''

pattern = re.compile('ABC\w\s+ (\S+)\s+\S+\s+(?=active)') # Positive look around
# So here ABC is first we are looking follwed by alphanumeric chr i.e 1,2 or any word then followed by bunch of  space
# then follwed by bunch of Non space chr , here we have taken (\S+) in group as we neeed to find only this part 
# then follweed by bunch of space and then follwed by active , so here it is looking for active but not capturing it

re.findall(pattern,string)
re.search(pattern,string) # The output shows that it is not capturing active but only taking it to look in pattern

#However we can also use non-captruing group
pattern = re.compile('ABC\w\s+ (\S+)\s+\S+\s+(?:active)')
re.findall(pattern,string)
re.search(pattern,string) # But here in output it is consuming active word, so if we want to specify some condition then

#use non-captruing group

#look aheads don't consume, non-capture group consumes
string ='abababacb'
#so we need to find wherever a's surronded by b , here we have 2 cases

pattern = re.compile('(?:b) (a) (?:b)')
re.findall(pattern,string)
#Here we did'nt get any result becoz when cursor starts from a and goes to b then there is no match as a is not surronded
# But when it goes for second occurance of a it should give the result as here 2nd a is surronded by b , but since
# cursor has already consume the 'b' at 2nd position for finding first a sp it starts from 2nd occurance of a
# and again when it starts from a there is no b behind 2nd occuance of a

#But when we give look around
pattern = re.compile('(?<=b)(a)(?=b)')
re.findall(pattern,string)

# Here it goes for 'a' and find 'b' behind and it fails but for 2nd occcurance of 'a' there b is behind and after a
# Here it does'nt consume first occurance of 'b'

####################################################################################################






























                                                   
                            

