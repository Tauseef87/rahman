'''
Problem Statement
Already, as of 2010, a quarter of Americans (24 percent) had posted product reviews or comments online, 
and 78 percent of internet users had gone online for product research. But those are ancient stats. 
Numbers are higher now. More recently, BrightLocal found in 2016 that 91 percent of consumers 
regularly or occasionally read online reviews, with 47 percent taking sentiment of local-business 
reviews — the tonality of a review’s text — into account in purchasing decisions. Breaking 
out the figures, 74 percent of consumers say that positive reviews make them trust a local 
business more, and 60 percent say that negative reviews make them not want to use a business, 
according to BrightLocal.

Customer reviews contain several forms of slient information. First there's the star rating, 
but ratings, even when broken out into categories - on Airbnb, for example, categories include 
accuracy, communication, cleanliness, location, check-in, and value - have zero explanatory power. 
So we have review text: free-form, voice-of-the-customer reactions. This text tells a story, 
and stories sell, so we need to know the aspects of a product or service discussed, the wording 
used to describe them, and the sentiment expressed.

Review text also reveals a lot about the reviewer, as Stanford University Prof. Dan Jurafsky 
explains in an exploration of review language, Natural Language Processing on Everyday Language.

So reviews are important, and the feelings expressed are key. To understand review content, 
including sentiment, at web and social scale, you need automated natural language processing (NLP) 
and other forms of AI


State of the Art
Commercial review-management platforms — from Bazaarvoice, PowerReviews, Yotpo, and others — 
help brands and online commerce sites collect reviews and redeploy them to boost sales.

'''

#Importing Dependences

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1500)
import itertools
import re
from nltk import word_tokenize,sent_tokenize
import requests
from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm
import os
os.chdir('F:/Review_Amazon')
#Data Collection

#link to Cornitos review on Amazon.in
links = 'https://www.amazon.in/Cornitos-Nachos-Crisps-Cheese-Herbs/product-reviews/B00GUOYZ7Y/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
response = requests.get(links)
response.status_code

#What is this status code?

#You might have came across Error 404-Not Found while browsing. That 404 is nothing but a status code returned by browser.

'''
Status codes
1xx     Informational   100-199
2xx     Success         200-299
3xx     Redirection     300-399
4xx     Client Error    400-499
5xx     Server Error    500-599
'''

response.content[:500]

#Apart from the content server sends back something called headers which has meta data, cookies and 
#other information.

response.headers

#It returns a dictionary

for key,value in response.headers.items():
    print(f'{key}:  {value}')

response.close()

'''
My real purpose of showing you the headers is to tell you that headers received using requests 
is different that from the your chrome browser. So servers can very easily track that you are 
using a script or Robotic Process Automation Tool, not a usual Browser.

'''

from fake_useragent import UserAgent
user_agent = UserAgent()

page = requests.get(links,headers={'user-agent':user_agent.chrome})

#We passed pre-defined headers into our requests, which will mimic the same header as chrome browser.

page

for k,v in page.headers.items():
    print(f'{k}: {v}')


#Now that we have response from the Amazon Server, Lets scrape data from it.
    
soup = BeautifulSoup(page.content,'html.parser')    

#Scraping Reviews

soup.findAll('span',{'data-hook':'review-body'})

#Trying to find all span in the page where data hook = review body
#If we give only find then only first occurance of span will be picked where data hook = review body
#Now all the reviews are inside the span with class=""

body = soup.findAll('span',{'data-hook':'review-body'})
[b.find('span').text for b in body]

#Greate we have all reviews, now lets grab rating

soup.findAll('i', {'data-hook' : "review-star-rating"})

rat = soup.findAll('i', {'data-hook' : "review-star-rating"})

[r.find('span').text for r in rat]

#Getting first number from the string.

[float(re.findall('(.*)(?= out)', r.find('span').text)[0]) for r in rat]

#The last thing we need to scrape is flavour name.

[a.text for a in soup.findAll('a',{'class':'a-size-mini a-link-normal a-color-secondary'})]

#Lets get everything after Flavour Name

[re.findall('(?<=Name: )(.*)',a.text)[0] for a in soup.findAll('a' , {'class':"a-size-mini a-link-normal a-color-secondary"})]

#Getting the number of reviews
num_review = soup.find('span',{'data-hook':'cr-filter-info-review-count'}).text
num_review = int(re.findall('(?<=of)(.*)(?=reviews)',num_review)[0])

'''
find -> find(tag,attributes) Both functions are used for fetching tag from HTML doc
The only difference is that find gets first occurance of tag while findall gets all
occurance of tag.

here we are finding all span where data hook = cr-filter-info-review-count

'''

#There are 339 reviews in total each page has 10 reviews, so that means we have iterate over 339/10 =33.9 ~ 34 pages

#Let instantiate three empty lists for reviews, ratings and flavour_name, to which we will append after every iteration over the pages.

reviews = []
ratings = []
flavour_name = []

'''
All the pages are in format
link + &pageNumber=1
link + &pageNumber=2
link + &pageNumber=3

'''

#The list.append method appends an object to the end of the list.
'''
>>> another_list = [1, 2, 3]
>>> my_list.append(another_list)
>>> my_list
['foo', 'bar', 'baz', [1, 2, 3]]
'''
# a list is an object. If you append another list onto a list, the first list will be a single object at the end of the list
#The list.extend method extends a list by appending elements from an iterable:
'''
>>> my_list
['foo', 'bar']
>>> another_list = [1, 2, 3]
>>> my_list.extend(another_list)
>>> my_list
['foo', 'bar', 1, 2, 3]
'''

    
for i in tqdm(range(int(num_review/10) + 1)):
    url = links + f'&pageNumber={i+1}'
    page = requests.get(url ,headers={'user-agent':user_agent.ie})
    soup = BeautifulSoup(page.content, 'html.parser')
    page.close()
    body = soup.findAll('span', {'data-hook' : "review-body"})
    reviews.extend([b.find('span').text for b in body])
    rat= soup.findAll('i', {'data-hook' : "review-star-rating"})
    ratings.extend([float(re.findall('(.*)(?= out)', r.find('span').text)[0]) for r in rat])
    flavour_name.extend([re.findall("(?<=Name: )(.*)", a.text)[0] for a in soup.findAll('a' , {'class':"a-size-mini a-link-normal a-color-secondary"})])
    sleep(10)    
#sleep(5) stops your program execution for 5 seconds.
    
len(reviews),len(ratings),len(flavour_name)

cornitos_df = pd.DataFrame({'reviews':reviews,'ratings':ratings,'flavour_name':flavour_name})
cornitos_df['Brand']='Cornitos'
cornitos_df = cornitos_df.drop_duplicates()
cornitos_df.head()

cornitos_df.to_csv('cornitos.csv',index=False)

#Lays

links = 'https://www.amazon.in/Lays-Potato-Chips-American-Style/product-reviews/B01CHWM75E/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
page = requests.get(links,headers={'user-agent':user_agent.ie})
soup = BeautifulSoup(page.content,'html.parser')
page.close()
num_reviews = soup.find('span',{'data-hook':'cr-filter-info-review-count'}).text
num_reviews = int(re.findall('(?<=of )(.*)(?= reviews)',num_reviews)[0])

'''
#Match

line = 'pet:cat i love cats'
#match(pattern,string,<flag=0>)
match = re.match(r'pet:\w\w\w',line)
#here \w means a single a-z alphabet or 0-9 numbers and _ with single  word character
print(match)
#Search and match are similar syntax
#Match function will search only at begining of string but search will search entire string
match = re.search(r'pet:\w\w\w',line)
print(match)
#Below code is for difference between match and search
line = 'i love cats pet:cat'
match = re.match(r'pet:\w\w\w',line)
print(match)
#Nothing has come because at start the pattern is not present
match = re.search(r'pet:\w\w\w',line)
print(match)
#Here the pattern found - it is the difference

#findall
line = 'pet:cat i love cats pet:cow i love cow'
#when we apply search then we will get first occurance of pet:
#But findall gives all occurance of pattern pet:
match = re.search(r'pet:\w\w\w',line)
print(match)

match = re.findall(r'pet:\w\w\w',line)
print(match)

'''



















reviews= []
ratings= []
flavour_name= []

for i in tqdm(range(int(num_review/10) + 1)):
    url = links + f'&pageNumber={i+1}'
    user_agent = UserAgent()
    page = requests.get(url ,headers={'user-agent':user_agent.msie})
    soup = BeautifulSoup(page.content, 'html.parser')
    body = soup.findAll('span', {'data-hook' : "review-body"})
    reviews.extend([b.find('span').text for b in body])
    rat= soup.findAll('i', {'data-hook' : "review-star-rating"})
    ratings.extend([float(re.findall('(.*)(?= out)', r.find('span').text)[0]) for r in rat])
    flavour_name.extend([re.findall("(?<=Name: )(.*)", a.text)[0] for a in soup.findAll('a' , {'class':"a-size-mini a-link-normal a-color-secondary"})])
    sleep(10) 
    page.close()

Lays_df = pd.DataFrame({'reviews':reviews,'ratings':ratings,'flavour_name':flavour_name})
Lays_df['Brand']='Lays'
Lays_df = Lays_df.drop_duplicates()
Lays_df.head()
Lays_df.to_csv('Lays.csv',index=False)

#Too Yum
links=['https://www.amazon.in/Too-Yumm-Multigrain-Chips-Chinese/product-reviews/B07B22428J/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
      'https://www.amazon.in/Too-Yumm-Multigrain-Chips-Papdi/product-reviews/B07B21TBQ7/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
      'https://www.amazon.in/Too-Yumm-Multigrain-Chips-Tomato/product-reviews/B07B21BFLF/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
      'https://www.amazon.in/TooYumm-Veggie-Stix-Chilly-Chataka/product-reviews/B075YP2RGM/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
      'https://www.amazon.in/Too-Yumm-Karare-Munchy-Masala/product-reviews/B07L3Q4998/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews']

reviews= []
ratings= []
flavour_name= []
for link in links:
    user_agent = UserAgent()
    page = requests.get( link ,headers={'user-agent':user_agent.random})
    soup = BeautifulSoup(page.content, 'html.parser')
    page.close()
    num_reviews= soup.find('span', {'data-hook': 'cr-filter-info-review-count'}).text
    num_reviews= int(re.findall("(?<=of )(.*)(?= reviews)", num_reviews)[0]) 
    for i in tqdm(range(int(num_reviews/10) + 1)):
        url = link + f'&pageNumber={i+1}'
        user_agent = UserAgent()
        page = requests.get(url ,headers={'user-agent':user_agent.ie})
        soup = BeautifulSoup(page.content, 'html.parser')
        body = soup.findAll('span', {'data-hook' : "review-body"})
        reviews.extend([b.find('span').text for b in body])
        rat= soup.findAll('i', {'data-hook' : "review-star-rating"})
        ratings.extend([float(re.findall('(.*)(?= out)', r.find('span').text)[0]) for r in rat])
        flavour_name.extend(re.findall('(?<=in/)(.*)(?=/product)', link) * len(rat))
        sleep(10)
        page.close()
    sleep(5)
    
TY_df = pd.DataFrame({'reviews':reviews,'ratings':ratings,'flavour_name':flavour_name})
TY_df['Brand']='Too_yum'
TY_df = TY_df.drop_duplicates()
TY_df.head()
TY_df.to_csv('TooYum.csv',index=False)

#Pringles
links= ['https://www.amazon.in/Pringles-Spanish-Jalapeno-Cheese-Flavour/product-reviews/B074N7B8Z9/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
        'https://www.amazon.in/Pringles-Potato-Chips-Cream-Onion/product-reviews/B01N5LQL5N/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
       'https://www.amazon.in/Kelloggs-Pringles-Potato-Chips-110g/product-reviews/B01H6QG78E/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
       'https://www.amazon.in/Pringles-South-African-Style-Flavour/product-reviews/B074ND86Y9/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews',
       'https://www.amazon.in/Pringles-Potato-Crisps-Pizza-flavour/product-reviews/B07D1CW55Q/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews']

reviews= []
ratings= []
flavour_name= []
for link in links:
    user_agent = UserAgent()
    page = requests.get( link ,headers={'user-agent':user_agent.random})
    soup = BeautifulSoup(page.content, 'html.parser')
    page.close()
    num_reviews= soup.find('span', {'data-hook': 'cr-filter-info-review-count'}).text
    num_reviews= int(re.findall("(?<=of )(.*)(?= reviews)", num_reviews)[0]) 
    for i in tqdm(range(int(num_reviews/10) + 1)):
        url = link + f'&pageNumber={i+1}'
        user_agent = UserAgent()
        page = requests.get(url ,headers={'user-agent':user_agent.ie})
        soup = BeautifulSoup(page.content, 'html.parser')
        body = soup.findAll('span', {'data-hook' : "review-body"})
        reviews.extend([b.find('span').text for b in body])
        rat= soup.findAll('i', {'data-hook' : "review-star-rating"})
        ratings.extend([float(re.findall('(.*)(?= out)', r.find('span').text)[0]) for r in rat])
        flavour_name.extend(re.findall('(?<=in/)(.*)(?=/product)', link) * len(rat))
        sleep(10)
        page.close()
    sleep(5)
    
Prin_df = pd.DataFrame({'reviews':reviews,'ratings':ratings,'flavour_name':flavour_name})
Prin_df['Brand']='Pringles'
Prin_df = Prin_df.drop_duplicates()
Prin_df.head()
Prin_df.to_csv('Pringles.csv',index=False)

#Importing Data

df1 = pd.read_csv('cornitos.csv')
df2 = pd.read_csv('Lays.csv')
df3 = pd.read_csv('Pringles.csv')
df4 = pd.read_csv('TooYum.csv')
df =  pd.DataFrame()
df = df.append([df1,df2,df3,df4])

df.head()

'''
Data Description
Features:
reviews: Reviews on the products by Buyers.
ratings: ratings associated with the reviews
flavour_name: Flavour of the Packaged Snacks
Brand: Brand of the Snacks(Cornitos, Too Yum, Lays, Pringles) We are considering these four you can 
collect more if you want.

'''

#Data Cleaning
df.isnull().sum()

df= df[~df.reviews.isnull()]

#Exploratory Data Analysis¶

#WordCount

df['word_count']=df.reviews.apply(lambda x: len(word_tokenize(str(x))))
df[['reviews','word_count']].head()

data = pd.DataFrame({'word_count':df.word_count,'label':df.Brand})
groups = data.label.unique().tolist()

#Pandas Index.tolist() function return a list of the values









































