# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:49:10 2015

@author: jward
"""

import pandas as pd
import numpy as np
import nltk
import re

#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |  PRE-PROCESS DATA   |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~


# helper function to clean tokens using nltk
def create_clean_tokens(s):    
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



#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      DATA EXPLORATION      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')
vect.fit(df.text)
all_features = vect.get_feature_names()

ham_dtm = vect.transform(df[df.spam==0].text)
ham_arr = ham_dtm.toarray()
del ham_dtm # to free up memory

spam_dtm = vect.transform(df[df.spam==1].text)
spam_arr = spam_dtm.toarray()
del spam_dtm, df # to free up memory

ham_counts = np.sum(ham_arr, axis=0)
del ham_arr # free up memory
spam_counts = np.sum(spam_arr, axis=0)
del spam_arr # free up memory

all_token_counts = pd.DataFrame({'token':all_features, 'ham':ham_counts, 'spam':spam_counts})
all_token_counts.sort('ham').tail(25)
all_token_counts.sort('ham').head(25)
all_token_counts.sort('spam').tail(25) # http is number 2 --> means emails that are links will be rejected; better way to deal with links?
all_token_counts.sort('spam').head(25)
# top words are often english stop words --> should filter out stop words
# repeat above code but remove stop words

# drop token counts and reload dataframe
del all_features, all_token_counts, ham_counts, spam_counts
df = pd.read_csv('../raw_data/email_text.csv', encoding='utf-8').dropna()
df['text'] = [create_clean_tokens(ii) for ii in df.text]




######################################
######################################
# |                                | #
# |   MODELS WITH STOP WORDS       | #
# |                                | #
######################################
######################################


#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      NAIVE BAYES      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, cross_val_score
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

# calc using full data set and cross_val_score
mn_nb = MultinomialNB()
vect = CountVectorizer()
X_fitted = vect.fit_transform(X)
scores_nb = cross_val_score(mn_nb, X_fitted, y, cv=10, scoring='roc_auc')
scores_nb.mean() # 0.98426





#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      LOGISTIC REGRESSION      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
scores_lr = cross_val_score(logreg, X_fitted, y, cv=10, scoring='roc_auc')
scores_lr.mean() # 0.99483

# what kind of emails is it missing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)

# vectorize text
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# fit model
logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)
y_pred = logreg.predict(X_test_dtm)

# where is the model missing?
confusion_matrix(y_test, y_pred) # [[4466,  134],
                                 #  [  60, 8149]])
# false positives
for fp in X_test[y_test < y_pred]:
    print fp, '\n'

# examples of false positives
'''
The following is an aerial photo of the WTC area It kinda brings on vertigo 
but is a phenomenal shot http

Bloody Hell 

Click here for more information about ETS The Next Generation http 

Save when you use our Customer Appreciation Spring Savings Certificate at Foot 
Locker Lady Foot Locker Kids Foot Locker and at our online stores Welcome to 
our Customer Appreciation Spring Savings Certificate Use the special 
certificate below and receive OFF your purchase ... 

Louise Your new pager number is Here are the instructions on sending messages 
Email for your SurePage pagenetmessage.net When you do n't have email but have 
internet access www.pagenet.com click on send a message Terminal click on arrow 
and a menu will drop down SurePage Enter your pin do not type Type your message 
Click on Send the Message 

Lokay bigfoot.com thought you would be interested in this article at Salon.com 
http It time to sit back and relax in luxury The all-new ES possesses the 
luxury amenities to pamper you and the cutting-edge technology to thrill you 
The all-new ES welcome to a new world of luxury Click To Enter A New World Of 
Luxury Your friend message I thought of you This Modern World By Tom Tomorrow 
http Sun Jan 

Copy do not forward this entire e-mail and paste it onto a new e-mail that you
will send Change all of the answers so that they apply to you Then send this 
to a whole bunch of people you know INCLUDING the person who sent it to you 
The theory is that you will learn a lot of little known facts about your friends 
Remember to send yours back to the person who sent it to you ...

One morning Dick Cheney and George W. Bush were having brunch at a restaurant 
The attractive waitress asks Cheney what he would like and he replies I have 
a bowl of oatmeal and some fruit And what can I get for you sir she asks 
George W. He replies How about a quickie  Why Mr. President the waitress says 
How rude you starting to act like Mr. Clinton and you have n't even been in 
office for a month yet As the waitress storms away Cheney leans over to Bush 
and whispers It pronounced quiche 

'''    

# false negatives
for fn in X_test[y_test > y_pred]:
    print fn, '\n'

'''
Hey What up Here the link that you requested I hope it what you needed Got ta 
run for now I be back on monday let me know if it helps Catch you later 
Complete your order here 

Re Refi for Hi Did you recieve my email from last week I happy to tell you 
that you can get a home loan at a rate Your tracking number is NG5132 You need 
to confirm your details within the next hours Please respond to this email 
address ecoquote03 yahoo.com Be sure to include the following Full name 
Phone Best time to reach you We will get back with you right away to discuss 
the details Best Regards Gladys Coley Ecoquote To never hear from us again 
powermarketing.capturehost.com/pak/powermarketing/gone.php 

New Page Dear Barclays member This email was sent by the Barclays server to 
verify your e-mail address.You must complete this process by clicking on the 
link below and entering your account information.This is done for your protection 
because some of our members are no longer have access to their onlyne access 
and we must verify it.To verify your e-mail address and access your bank account 
click on the lick below https Please fill in the required information 
This is required for us to continue to offer you a safe and risk free environment 
Thanks you Accounts Management 

ATTENTION Latest updates Download any latest software now http Marian with regards 

This is b3tt3r th3n Viagr4 Dear Valued Customer Today you are about to learn 
about the future The future in Prescript1on buying Right now in Canada they use 
almost ALL Geneeric drugs to cut back on spending and they probably spend 
about a of what the USA spends on Prescript1on medications Today is your chance 
to get in on these S4vings 

'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# |   INITIAL TAKEAWAYS   |
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The spam filter performs well (98%+ accuracy) but has too many false positives
# coworkers spam each other -- if I kept "replies" and "forwarded by" I could improve this ... maybe?




#########################################
#########################################
# |                                   | #
# |   MODELS WITHOUT STOP WORDS       | #
# |                                   | #
#########################################
#########################################


#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      NAIVE BAYES      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# create features and columns
X = df.text
y = df.spam

# create train_test_split datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)

# fit and transform X_train, and transform X_test
vect = CountVectorizer(stop_words='english')
train_dtm = vect.fit_transform(X_train)
test_dtm = vect.transform(X_test)

# run Multinomial Naive Bayes
mn_nb = MultinomialNB()
mn_nb.fit(train_dtm, y_train)

# make predictions
y_pred = mn_nb.predict(test_dtm)
accuracy_score(y_test, y_pred)      # 0.98594
confusion_matrix(y_test, y_pred)    # [[4496,  104],
                                    #  [  76, 8133]])

# calc predict probability for roc_auc score
y_prob = mn_nb.predict_proba(test_dtm)[:, 1]
roc_auc_score(y_test, y_prob) # 0.996839

# calc using full data set and cross_val_score
mn_nb = MultinomialNB()
X_fitted = vect.fit_transform(X)
scores_nb = cross_val_score(mn_nb, X_fitted, y, cv=10, scoring='roc_auc')
scores_nb.mean() # 0.98780





#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      LOGISTIC REGRESSION      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
scores_lr = cross_val_score(logreg, X_fitted, y, cv=10, scoring='roc_auc')
scores_lr.mean() # 0.994938

# what kind of emails is it missing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)

# vectorize text
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# fit model
logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)
y_pred = logreg.predict(X_test_dtm)

# where is the model missing?
confusion_matrix(y_test, y_pred) # [[4460,  140],
                                 #  [  60, 8149]]
# false positives
for fp in X_test[y_test < y_pred]:
    print fp, '\n'

# examples of false positives
'''


'''    

# false negatives
for fn in X_test[y_test > y_pred]:
    print fn, '\n'

'''


'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~
# |       TAKEAWAYS       |
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  HOW DO I TUNE NB AWAY FROM FALSE POSITIVES?



