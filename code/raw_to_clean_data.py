# -*- coding: utf-8 -*-

'''
### ~~> SET WORKING DIRECTORY TO THE 'code' FOLDER OF THIS PROJECT
'''

# Set constants
DATA_PATH = '../raw_data/raw_email_inventory.csv'
#DATA_PATH = '../raw_data/scratch_test_sample.csv'


# import modules
from bs4 import BeautifulSoup
import email
    

def separate_header_body(msg):
    # parse email into headers and body (i.e., metadata and data)
    parser = email.parser.Parser()
    temp = parser.parsestr(msg.as_string())
    headers = temp.items()
    if not temp.is_multipart():
        body = unicode(temp.get_payload(), 'utf-8', errors='ignore')
    else:
        body = unicode(temp.get_payload()[0].as_string(), 'utf-8', errors='ignore')
    return headers, body

def strip_html(html):
    # strip any html and remove returns (\r), newlines (\n), tabs, etc...    
    b = BeautifulSoup(html)
    txt = b.get_text()
    txt = txt.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').replace('\\', '').replace('=20', '')
    return txt
    
def parse_email(s, ham_spam, remove_replies=True):
    header, body = separate_header_body(s)
    if ham_spam == 'ham':
        # get rid of problem email which crashes python
        if header[3][1].lower().find('enron mentions') == -1:
            try:    
                body = strip_html(body)
                if remove_replies:
                    reply = body.find('-----Original Message-----',0)
                    forward = body.find('----------',0)
                    if reply != -1 or forward != -1:
                        if reply != -1 and forward != -1:
                            if reply < forward:
                                cutoff = reply
                            else:
                                cutoff = forward
                        elif reply != -1 and forward == -1:
                            cutoff = reply
                        else:
                            cutoff = forward
                        body = body[:cutoff]
                #email_headers.append(header)
                email_text.append(body)
                return True
            except:
                return False
    else:
        try:    
            body = strip_html(body)
            if remove_replies:
                missed_header = body.find('Content-Type: text/html Content-Transfer-Encoding: \7bit')
                if missed_header != -1:
                    body = body[missed_header + 55:]
            #email_headers.append(header)
            email_text.append(body)
            return True
        except:
            return False
        


# get list of emails to parse
with open(DATA_PATH, 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line


email_types = []
email_headers = []
email_text = []
skips = ['../raw_data/ham/kitchen-l/_americas_esvl/691.txt', \
         ]
problems = []

for msg in list_of_emails:
    
    print msg
    if msg not in skips:  
     # open email as message from file using the email module
        with open(msg, 'rU') as f:
            content = email.message_from_file(f)
        
        # is email ham or spam?
        ham_or_spam = msg[12:15]
        
        # call parse_email() to get email header and body
        success = parse_email(content, ham_or_spam)
        
        # store whether the email is ham or spam
        if success:            
            if ham_or_spam == 'ham':
                email_types.append(0)
            else:
                email_types.append(1)
        else:
            problems.append(msg)


# create dataframe .... 
import pandas as pd
df = pd.DataFrame(zip(email_types, email_text), columns=['spam', 'text'])

# save to csv file
df.to_csv('../raw_data/email_text.csv', encoding='utf-8', index=False)
