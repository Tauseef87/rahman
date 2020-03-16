import requests
# request is an api in python
response = requests.get('http://www.google.com/')
help(requests.get)
response.status_code
#Status Code is something like 404 - Not found Error returned by server
#Status code example -  100-199 Informational,Success - 200-299
#Redirection - 300-399, Client error - 400-499,Server Error - 500-599
response.content[:2000]
#It displays the raw html file.
#The output of raw file is coming in byte code character eg b'<!doctype
#By default python is unicode of UTF-32
#b' mainly used to becoz when you request in server they try to compress it
#The byte code can be decoded to utf. content.decode(utf)
import sys
print(sys.getsizeof('hello'))
print(sys.getsizeof(b'hello'))

# HTTP Headers  --> It is the code that transfers data between web server and browser.
# Every time a request is send , it is send as request header and every time a response is done
# it is done with response header
response.headers

for ky,val in response.headers.items():
    print(f'{ky:{17}}: {val}')
# f ' -> format string {17} it is nothing to print after 7 spaces   
# HTTP request Header
# Type,capabilities and verson of browser that generate the request
# OS by client,Various type of output eg html,json
# when we do scrapping by the header type the site may recognize that it is done by script
# or server can easily track that it is a RPA or Script not a browser
# To overcome this we use fake user agent
# Fake user agent will send the request like browser type but it is a mimic of scriipt of python   
   
response.close()    

#Fake agent
from fake_useragent import UserAgent    
user_agent= UserAgent()

page = requests.get('https://www.google.com/',headers={'user-agent':user_agent.chrome} )
#Here the request header will mimic the python but very similar same
#Server will assume that it is using chrome browser not any script
# Different type of browser -> user_agent.ie,user_agent.opera,user_agent.google,user_agent.firefox
page.status_code
page.close()

#Beautiful Soup - > if you have to retrive data from html file in much attractive file, it creates
# parser for html and xml. It is used in web scrapping but not specific to web scrap.
# %%writefile - Magic function
%%writefile example.html
<!DOCTYPE html> # It is doctype as html
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NLP 100 Hours Batch II</title>
    </head>
    <body>
        <div>
            <p>In the first Division Tag</p>
        </div>
        <div>
            <p>In the Second Division Tag</p>
        </div>
    </body>
</html>

with open('example.html', 'r') as f:
    text= f.read()
   
print(text)
from bs4 import BeautifulSoup
soup = BeautifulSoup(text)
print(soup.prettify())
#Prettify -> It makes tree structure of html file. It gives indentation
# Html Structure - Html -> Head -> Title,Html ->Body->DIV->P
soup.title
#title -> to access the title from html(It will show the first title if you have more than 1 title)
soup.title.text
soup.body.text 
#body - it gives the all data present in body of html
soup.find('p').text
# find('p') - It parses the whole html file and will give data inside the first p of html body

for p in soup.find_all('p'):
    print(p.text)
# find_all -    It parses the whole html file and will give data inside the  p of html body
for child in soup.body.children:
    print(child)  
# two div are there which is child of body  tag  
for child in soup.head.descendants:
    print(child)    
# if we click on body html is descendant   
soup.div.parent.name    
# Body is the parent of Div
soup.div.next_sibling.next_sibling
# It will give next sibling of div
# Scraping amazon Reviews
from fake_useragent import UserAgent
user_agent = UserAgent()

link= 'https://www.amazon.in/Apple-iPhone-Pro-Max-256GB/product-reviews/B07XVMDRZW/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'
page = requests.get(link,headers={'user-agent':user_agent.chrome} )
page.status_code
soup = BeautifulSoup(page.content)
page.close()
review_body = soup.find_all('span',{'data-hook':"review-body"})
soup.find_all('span',{'data-hook':"review-body"})[0]
# or soup.find_all('span',{'class':"a-size-base review-text review-text-content"})
for b in review_body:
    print(b.find('span').text)
    print('--------')
reviews= [b.find('span').text for b in review_body]    
reviews[0]
print(type(review_body))

rat = soup.findAll('i', {'data-hook' : "review-star-rating"})
[r.find('span').text for r in rat]
ratings= [r.find('span').text for r in rat]
import re
[re.findall('(.*)(?= out)', r) for r in ratings]
# here we extracting only that part of string which exclude "out of" word 
from time import sleep

reviews= []
ratings= []
# Here range = 14 becoz the page contains 10 review per page and there are 140 reviews
for i in range(14):
    link= f'https://www.amazon.in/Apple-iPhone-Pro-Max-256GB/product-reviews/B07XVMDRZW/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={i+1}'
    page = requests.get(link ,headers={'user-agent':user_agent.chrome})
    soup = BeautifulSoup(page.content)
    page.close()
    body = soup.findAll('span', {'data-hook' : "review-body"})
    reviews.extend([b.find('span').text for b in body])
    rat= soup.findAll('i', {'data-hook' : "review-star-rating"})
    ratings.extend([float(re.findall('(.*)(?= out)', r.find('span').text)[0]) for r in rat])
    sleep(5)
    
len(reviews), len(ratings)    

import pandas as pd
final_df= pd.DataFrame({'ratings': ratings, 'reviews' : reviews})
