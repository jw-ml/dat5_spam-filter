# -*- coding: utf-8 -*-

'''
### ~~> SET WORKING DIRECTORY TO THE 'code' FOLDER OF THIS PROJECT
'''

# Set constants
#DATA_PATH = '../preprocessed_data/preprocessed_email_inventory.csv'
DATA_PATH = '../raw_data/raw_email_inventory.csv'


# import modules
from bs4 import BeautifulSoup
import nltk
import re
import email

def separate_header_body(msg):
    # parse email into headers and body (i.e., metadata and data)
    parser = email.parser.Parser()
    temp = parser.parsestr(msg.as_string())
    headers = temp.items()
    body = unicode(temp.get_payload())
    return headers, body

def strip_html(html):
    # strip any html and remove returns (\r), newlines (\n), tabs, etc...    
    b = BeautifulSoup(html)
    txt = b.get_text()
    txt = txt.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').replace('\\', '').replace('=20', '')
    return txt
    
def parse_email(msg):
    header, body = separate_header_body(msg)
    body = strip_html(body)
    email_headers.append(header)
    email_text.append(body)

# get list of emails to parse
with open(DATA_PATH, 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line


email_types = []
email_headers = []
email_text = []

for msg in list_of_emails[:15]:
    # open email as message from file using the email module
    with open(msg, 'rU') as f:
        content = email.message_from_file(f)
    # store whether the email is ham or spam
    ham_or_spam = msg[12:15]
    if ham_or_spam == 'ham':
        email_types.append(0)
    else:
        email_types.append(1)
    # call parse_email() to get email header and body
    parse_email(content)


# create dataframe .... 
import pandas as pd
df = pd.DataFrame(zip(email_types[0:10], email_text[0:10]), columns=['spam', 'text'])










#####################################
txt = 'this is a temp string'

# Tokenize into sentences
sentences = []
for sent in nltk.sent_tokenize(txt):
    sentences.append(sent)
sentences[:10]


# Tokenize into words
tokens = []
for word in nltk.word_tokenize(txt):
    tokens.append(word)

clean_tokens = [token for token in tokens if re.search('^[$a-zA-Z]+', token)]
clean_tokens[:100]# Tokenize into words
