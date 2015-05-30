# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:01:40 2015

@author: jward
"""

# import modules

email_types = []
email_text = []


# Set program constants; USE TO PICK SAMPLE OR FULL DATA
DATA_PATH = '../raw_data/raw_data_inventory.csv'
#DATA_PATH = '../raw_data/raw_data_sample.csv'  

# get list of emails
with open(DATA_PATH, 'rU') as f:
    list_of_emails = [row[:-1] for row in f] # have to remove the '\n' at the end of each line

# run through list of emails and parse all messages into header and body
for msg in list_of_emails:
    
        # open email as message from file using the email module
    with open(msg, 'rU') as f:
        content = f.read().decode('utf-8', errors='ignore')
    
    ham_or_spam = msg[12:15]
    if ham_or_spam == 'ham':
        email_types.append('ham')
    else:
        email_types.append('spam')
    
    email_text.append(content)
    
# create dataframe 
import pandas as pd
df = pd.DataFrame(zip(email_types, email_text), columns=['spam', 'full_text'])


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


X = df.full_text
y = df.spam.map({'ham':0, 'spam':1})
for k in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=k)
    
    vect = CountVectorizer()
    train_dtm = vect.fit_transform(X_train)
    test_dtm = vect.transform(X_test)
    
    nb = MultinomialNB()
    nb.fit(train_dtm, y_train)
    y_pred = nb.predict(test_dtm)
    accuracy_score(y_test, y_pred) # 0.99838
    
#results
#    0.99869421614563325
#    0.99846378370074507
#    0.99877102696059605
#    0.99831016207081957
#    0.99815654044089408
#    0.99807972962593128
#    0.99900145940548424
#    0.99854059451570776
#    0.99915508103540973
#    0.99846378370074507

confusion_matrix(y_test, y_pred)
#array([[4769,    4],
#       [  16, 8230]])

# false positives
for subj in X_test[y_test < y_pred]:
        print subj, '\n'

# false negatives
for subj in X_test[y_test > y_pred]:
    print subj, '\n'







confusion_matrix(y_test, y_pred)



