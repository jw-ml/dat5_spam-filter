# -*- coding: utf-8 -*-

import string
from bs4 import BeautifulSoup
# from sklearn.feature_extraction.text import CountVectorizer

'''
def remove_html_tags(all_text):
    soup = BeautifulSoup(all_text)
    txt = soup.get_text(" ", strip=True)
    return txt
'''

def parse_email(f, email_type):
    f.seek(0)    
    all_text = f.read()
    
    if email_type == 'ham':
        # split off metadata        
        # content = all_text.split("X-FileName:")
    else:
        # remove any html tags
        all_text = remove_html_tags(all_text)
        # split off metadata
        content = all_text.split("Content-Transfer-Encoding:")
    
    # create empty string
    words = ""

    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
    
        # extract words
        list_of_words = []
        text_string = text_string.replace('\n',' ')
        text_string = text_string.replace('\t',' ')
        text_string = text_string.split(" ")
        for word in text_string:
            if word != '':
                list_of_words.append(word)
        words = ' '.join(list_of_words)
        return words


# get list of emails to parse
with open('../raw_data/scratch_test_sample.csv', 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line

parsed_emails = []
email_types = []
for email in list_of_emails[:20]:
    with open(email, 'rU') as f:
        if email[12:16] == 'ham':
            text = parse_email(f, 'ham')
            parsed_emails.append(text)
            email_types.append('ham')
        else:
            text = parse_email(f, 'spam')
            parsed_emails.append(text)
            email_types.append('spam')
