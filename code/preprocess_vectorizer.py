# -*- coding: utf-8 -*-

import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# separate each email into a string of words free of punctuation and other 'noise'
def parse_email(f):
    
    f.seek(0)    
    content = f.read()
    
    # create empty string
    words = ""
    
    ### remove punctuation
    text_string = content.translate(string.maketrans("", ""), string.punctuation)
    
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
with open('../preprocessed_data/prepro_email_inventory.csv', 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line

# remove any .DS_Store entries
list_of_emails = [row for row in list_of_emails if row.find('.DS_Store') == -1]

parsed_emails = []
email_types = []
for email in list_of_emails:
    with open(email, 'rU') as f:
        mrkr = email.rfind('/')        
        spam_or_ham = email[mrkr-3:mrkr]
        if spam_or_ham == 'ham':
            text = parse_email(f)
            parsed_emails.append(text)
            email_types.append('ham')
        else:
            text = parse_email(f)
            parsed_emails.append(text)
            email_types.append('spam')
            



vectorizer = CountVectorizer(min_df=1, decode_error='ignore', stop_words='english')
X = vectorizer.fit_transform(parsed_emails)
col_names = vectorizer.get_feature_names()
word_matrix = X.toarray()

data = pd.DataFrame(word_matrix, columns=col_names)
response = pd.DataFrame(email_types, columns=['Response'])

