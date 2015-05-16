# -*- coding: utf-8 -*-

'''
### ~~> SET WORKING DIRECTORY TO THE 'code' FOLDER OF THIS PROJECT
'''


# import modules
from bs4 import BeautifulSoup
import email


# Set program constants
DATA_PATH = '../raw_data/raw_data_inventory.csv'
#DATA_PATH = '../raw_data/raw_data_sample.csv'


# declare variables
email_types = []
email_headers = []
email_text = []
problems = []
skips = ['../raw_data/ham/kitchen-l/_americas_esvl/691.txt']
problem_headers = [ 'Content-Type: text/html Content-Transfer-Encoding: \7bit', \
                    'Content-type: multipart/alternative; boundary="------------APBB482139743837-1"  --------------APBB482139743837-1 Content-type: text/plain; charset=us-ascii Content-Transfer-Encoding: \7bit', \
                    'Content-Type: text/plain;  charset="us-ascii" Content-Transfer-Encoding: quoted-printable', \
                    'Content-Type: text/plain; charset="iso-8859-1" Content-Transfer-Encoding: base64', \
                    'Content-Type: text/plain;  charset="iso-8859-1" Content-Transfer-Encoding: \7bit', \
                    'Content-Type: text/plain; charset="iso-8859-1" Content-Transfer-Encoding: \7bit', \
                    'Content-Type: text/plain;  charset="Windows-1252" Content-Transfer-Encoding: quoted-printable']

   
# create functions

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

def find_problem_header(body):
    result = -1
    head_len = 0
    at_least_one = False
    for ph in problem_headers:
        temp = body.find(ph)
        if temp > -1 and not at_least_one:
            result = temp
            head_len = len(ph)
            at_least_one = True
        if temp < result and at_least_one:
            result = temp
            head_len = len(ph)
    return result, head_len

def get_cutoff(reply, forward):
    if reply == -1 and forward == -1:
        return None
    elif reply == -1 and forward > -1:
        return forward
    elif reply > -1 and forward == -1:
        return reply
    else:
        if reply < forward:
            return forward
        else:
            return reply

def parse_ham(header, body, remove_replies):
    if header[3][1].lower().find('enron mentions') == -1: # get rid of problem emails
        try:    
            body = strip_html(body)
            if remove_replies:
                reply = body.find('-----Original Message-----',0)
                forward = body.find('-------',0)
                cutoff = get_cutoff(reply, forward)
                body = body[:cutoff]
            #email_headers.append(header)
            email_text.append(body)
            email_types.append('ham')
            return True
        except:
            return False   

def parse_spam(header, body, remove_replies):
    try:    
        body = strip_html(body)
        if remove_replies:
            missed_header, head_len = find_problem_header(body)
            if missed_header > -1:
                body = body[missed_header + head_len:]
        #email_headers.append(header)
        email_text.append(body)
        email_types.append('spam')
        return True
    except:
        return False

    
def parse_email(s, ham_spam, remove_replies=True):
    header, body = separate_header_body(s)
    if ham_spam == 'ham':
        is_success = parse_ham(header, body, remove_replies)
        return is_success
    else:
        is_success = parse_spam(header, body, remove_replies)
        return is_success

        

# get list of emails to parse
with open(DATA_PATH, 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line


for msg in list_of_emails:
    
    if msg not in skips:  
     # open email as message from file using the email module
        with open(msg, 'rU') as f:
            content = email.message_from_file(f)
        # is email ham or spam?
        ham_or_spam = msg[12:15]
        # call parse_email() to get email header and body
        success = parse_email(content, ham_or_spam)
        # store emails that cannot be parsed
        if not success:            
            problems.append(msg)


# create dataframe .... 
import pandas as pd
df = pd.DataFrame(zip(email_types, email_text), columns=['spam', 'text'])

# save to csv file
df.to_csv('../raw_data/email_text_150515.csv', encoding='utf-8', index=False)
