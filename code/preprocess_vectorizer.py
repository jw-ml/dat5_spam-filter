# -*- coding: utf-8 -*-

''' make sure your working directory is set to the 'code' subdirectly '''

import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# constants
DATA_PATH = '../preprocessed_data/preprocessed_email_inventory.csv'
#DATA_PATH = '../preprocessed_data/sample_data_inventory.csv'

# separate each email into a string of words free of punctuation and other 'noise'
def parse_email(f):
    
    # make sure to start at beginning of file; read contents of file
    f.seek(0)    
    content = f.read()
    
    # create empty string
    words = ""
    
    # remove punctuation
    text_string = content.translate(string.maketrans("", ""), string.punctuation)
    
    # extract words
    list_of_words = []
    text_string = text_string.replace('\n',' ') # remove line breaks
    text_string = text_string.replace('\t',' ') # remove tabs
    text_string = text_string.split(" ")        # each space (' ') creates a new list item
    for word in text_string:
        # filter out empty entries with if statement
        if word != '':
            list_of_words.append(word)
    words = ' '.join(list_of_words)
    return words


# get list of emails to parse
with open(DATA_PATH, 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line

parsed_emails = []  # for email text
email_types = []    # for response array (i.e., ham vs spam)

for email in list_of_emails:
    with open(email, 'rU') as f:
        mrkr = email.rfind('.txt')          # used to identify if email is ham or spam     
        spam_or_ham = email[mrkr-3:mrkr]    # grab "ham" or s"pam"
        if spam_or_ham == 'ham':
            text = parse_email(f)       # clean email into distinct words
            parsed_emails.append(text)  # add clean email to parsed_email list
            email_types.append('ham')   # add appropriate entry to Response array
        else:
            text = parse_email(f)
            parsed_emails.append(text)
            email_types.append('spam')

# Use sklearn to count each word in each email and create the appropriate feature matrix
# ~~> instantiate CountVectorizer object; only counts words with more than one character   
vectorizer = CountVectorizer(min_df=1, decode_error='ignore', stop_words='english')
word_matrix = vectorizer.fit_transform(parsed_emails)
col_names = vectorizer.get_feature_names()
X = word_matrix.toarray()   # features

# creates X and y
X = pd.DataFrame(X, columns=col_names)
y = pd.DataFrame(email_types, columns=['Response'])

# creates dataframe for exploration (DOESN"T WORK )
#df = pd.merge(y, X, how='left', left_index=True, right_index=True) # get stuck when run


