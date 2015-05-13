# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:49:10 2015

@author: jward
"""

import pandas as pd
import nltk
import re

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




