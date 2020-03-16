#Task 1
abbr = 'NLP'
full_text = 'Natural Language Processing'
print(abbr +' Stands for '+full_text)  
print(abbr ,' Stands for ',full_text)  

#Task2
def repeat(n):
    str = 'NLP'
    print(str*n)
repeat(5)    

#task3
string = "04-August-2019"
print("Day =",string[:2]," Month = ",string[3:9]," Year = ",string[-4:] )

#task4
string = "SSuuppeerrvviisseeddlleeaarrnniinng"
print(string[0::2])

#task5
my_file = open('F:\Algorithmica\MyCodes\owlcreek.txt')
content = my_file.read()
len(content)
my_file.close()

#task6
New_content = content.replace('.  ','. ')
with open("F:\\Algorithmica\\MyCodes\\new_owlcreek.txt",'w') as f:
    f.write(New_content)

#task7
import re
print(re.search('^man$',content))

str = ''' Hello Everyone
          Welcome to Python
          '''
print(re.search(r'^H$',str))         
