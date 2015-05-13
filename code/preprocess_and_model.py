# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:49:10 2015

@author: jward
"""

import pandas as pd
import nltk
import re

#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |  PRE-PROCESS DATA   |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~


# helper function to clean tokens using nltk
def create_clean_tokens(s):    
    print ii
    tokens = []
    for word in nltk.word_tokenize(s):
        tokens.append(word)
    clean_tokens = [token for token in tokens if re.search('^[$a-zA-Z]+', token)] # to fix: regular expression changes '$500' to '$'; want '$500' but not 5000
    to_return = ' '.join(clean_tokens)
    return to_return

# load the data
df = pd.read_csv('../raw_data/email_text.csv', encoding='utf-8')
df = df.dropna() # to fix: don't put these in the sample to begin with...

# clean the emails into a string of clean tokens
df['text'] = [create_clean_tokens(ii) for ii in df.text]


#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      NAIVE BAYES      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# create features and columns
X = df.text
y = df.spam

# create train_test_split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)

# fit and transform X_train, and transform X_test
vect = CountVectorizer()
train_dtm = vect.fit_transform(X_train)
test_dtm = vect.transform(X_test)

# run Multinomial Naive Bayes
mn_nb = MultinomialNB()
mn_nb.fit(train_dtm, y_train)

# make predictions
y_pred = mn_nb.predict(test_dtm)
accuracy_score(y_test, y_pred)      # first run: 0.986259
confusion_matrix(y_test, y_pred)    # array([[4586,   96],
                                    #        [  80, 8047]])


# calc predict probability for roc_auc score
y_prob = mn_nb.predict_proba(test_dtm)[:, 1]
roc_auc_score(y_test, y_prob) # first run: 0.99747




