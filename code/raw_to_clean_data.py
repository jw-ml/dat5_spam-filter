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
subjects = []
timestamps = []
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
    '''
    Takes a full email message and separates it into the email's header and its body.
    Returns the header and the body (i.e., metadata and text data)
    '''
    parser = email.parser.Parser()
    temp = parser.parsestr(msg.as_string())
    header = dict(temp.items()) # creates a list of tuples, then converts to dictionary
    # look for subject    
    try:    
        subject = unicode(header['Subject'], 'utf-8', errors='ignore')
    except:
        subject = None
    # look for timestamp
    try:    
        timestamp = unicode(header['Date'], 'utf-8', errors='ignore')
    except:
        try:
            timestamp = unicode(header['Received'], 'utf-8', errors='ignore')
        except:
            timestamp = None
    # check if email is multipart. If Yes, only return the first part
    # ~~> This is probably a big area for improvement. Relates to mainly to Spam messages in the sample.
    if not temp.is_multipart():
        body = unicode(temp.get_payload(), 'utf-8', errors='ignore')
    else:
        body = unicode(temp.get_payload()[0].as_string(), 'utf-8', errors='ignore')
    return timestamp, subject, body



def strip_html(html):
    '''
    Takes a string of text and strips any html tags that might be included.
    It also strips out special elements such as returns (\r), newlines (\n), tabs (\t), etc.
    Returns text string
    '''
    b = BeautifulSoup(html)
    txt = b.get_text()
    txt = txt.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').replace('\\', '').replace('=20', '')
    return txt



def find_problem_header(body):
    '''
    Some of the spam email have problem header elements that are not caught in \
    the separate_header_body function. 
    Takes the text from email body
    Returns the location of the problem header and the length of the header
    '''
    # declare variables result, header length (head_len), and whether there is more than one problem header
    result = -1
    head_len = 0
    at_least_one = False
    # see if any of the identified problem headers are in the email
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
    '''
    Determines whether a reply or forwarded response comes first in the email string.
    Takes the reply location and the forward location as inputs.
    Returns 
    '''
    if reply == -1 and forward == -1:
        return None
    elif reply != -1 and forward != -1:
        return min(reply, forward) # get first occurence
    else:
        return max(reply, forward) # get non -1 occurence



def parse_ham(timestamp, subject, body, remove_replies):
    if subject.lower().find('enron mentions') == -1: # get rid of problem emails
        try:    
            body = strip_html(body)
            if remove_replies:
                reply = body.find('-----Original Message-----',0)
                forward = body.find('-------',0)
                cutoff = get_cutoff(reply, forward)
                body = body[:cutoff]
            timestamps.append(timestamp)
            subjects.append(subject)
            email_text.append(body)
            email_types.append('ham')
            return True
        except:
            return False   

def parse_spam(timestamp, subject, body, remove_replies):
    try:    
        body = strip_html(body)
        if remove_replies:
            missed_header, head_len = find_problem_header(body)
            if missed_header > -1:
                body = body[missed_header + head_len:]
        timestamps.append(timestamp)
        subjects.append(subject)
        email_text.append(body)
        email_types.append('spam')
        return True
    except:
        return False

    
def parse_email(s, ham_spam, remove_replies=True):
    '''
    Processes the raw email into text that can be easily parsed.
    '''
    timestamp, subject, body = separate_header_body(s)
    if ham_spam == 'ham':
        is_success = parse_ham(timestamp, subject, body, remove_replies)
        return is_success
    else:
        is_success = parse_spam(timestamp, subject, body, remove_replies)
        return is_success

        

# get list of emails to parse
with open(DATA_PATH, 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line


# run through list of emails and parse all messages into header and body
for msg in list_of_emails:
    
    if msg not in skips:  # temporary 'if-statement'; used to deal with emails that crashed python
        # open email as message from file using the email module
        with open(msg, 'rU') as f:
            content = email.message_from_file(f)
        if not content.is_multipart():
            # is email ham or spam?
            ham_or_spam = msg[12:15]
            # call parse_email() to get email header and body
            success = parse_email(content, ham_or_spam)
            # store emails that cannot be parsed
            if not success:            
                problems.append(msg)


# create dataframe 
import pandas as pd
df = pd.DataFrame(zip(email_types, timestamps, subjects, email_text), columns=['spam', 'timestamp', 'subject', 'text'])

# save to csv file
df.to_csv('../raw_data/email_text_150516.csv', encoding='utf-8', index=False)
