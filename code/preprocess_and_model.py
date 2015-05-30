# -*- coding: utf-8 -*-
"""
@author: jward
"""


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  |       IMPORT MODULES       |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

#data_file = '../raw_data/email_text_150512.csv'
#data_file = '../raw_data/email_text_150515.csv'
#data_file = '../raw_data/email_text_150516.csv'
data_file = '../raw_data/email_text_150516_withRF.csv'


#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |  PRE-PROCESS DATA   |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~


# function to clean tokens using nltk
def create_clean_tokens(s):   
    try: # to deal with null 'subject' observations
        tokens = []
        for word in nltk.word_tokenize(s):
            word = word
            tokens.append(word)
        clean_tokens = [token for token in tokens if re.search('^[$a-zA-Z]+', token)] # to fix: regular expression changes '$500' to '$'; want '$500' but not 5000
        to_return = ' '.join(clean_tokens)
        return to_return
    except: # if above fails, return null value
        return s

# load the data
df = pd.read_csv(data_file, encoding='utf-8')

# look at null values
df.isnull().sum()
df = df.dropna().reset_index(drop=True) # need to find a better way of dealing with this; most likely in the steps cleaning the data...
#df = df[df.text.isnull()==False].reset_index(drop=True)

# clean the emails into a string of clean tokens
df['spam'] = df.spam.map({'ham':0, 'spam':1})
df['text'] = [create_clean_tokens(ii) for ii in df.text]
df['subject'] = [create_clean_tokens(ii) for ii in df.subject]



#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      DATA EXPLORATION      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


vect = CountVectorizer(stop_words='english')
vect.fit(df.text)
all_features = vect.get_feature_names()

ham_dtm = vect.transform(df[df.spam==0].text)
ham_arr = ham_dtm.toarray()
del ham_dtm # to free up memory
ham_arr.shape

spam_dtm = vect.transform(df[df.spam==1].text)
spam_arr = spam_dtm.toarray()
del spam_dtm, df # to free up memory
spam_arr.shape

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
df = pd.read_csv(data_file, encoding='utf-8').dropna()
df['text'] = [create_clean_tokens(ii) for ii in df.text]


####################################################
####################################################
# |                                              | #
# |              PROGRAM FUNCTIONS               | #
# |                                              | #
####################################################
####################################################


def get_model_data(data, feature_col, random_state_num):
    '''creates features and response, then returns training and test data'''
    X = data[feature_col]
    y = df.spam
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state_num)
    return X_train, X_test, y_train, y_test
    
    
def transform_data_to_dtm(train, test, remove_stop_words=True):
    '''converts text to data-term-matrices'''
    if remove_stop_words:    
        vect = CountVectorizer(stop_words='english')
    else:
        vect = CountVectorizer(stop_words=None)
    train_dtm = vect.fit_transform(X_train)
    test_dtm = vect.transform(X_test)
    return train_dtm, test_dtm


def run_multinomial_nb(X_tr, X_tst, y_tr, y_tst, pred_prob=False):
    '''Returns accuracy_score and predictions'''
    #mn_nb = MultinomialNB(class_prior=[0.75, 0.25])
    mn_nb = MultinomialNB()    
    mn_nb.fit(X_tr, y_train)
    if not pred_prob:
        # make predictions
        y_pred = mn_nb.predict(X_tst)
        acc = accuracy_score(y_tst, y_pred)
        return acc, y_pred
    else:
        # calc predict probability for roc_auc score
        y_prob = mn_nb.predict_proba(X_tst)[:, 1]
        roc = roc_auc_score(y_tst, y_prob)
        return roc, y_prob
        
def run_log_reg(X_tr, X_tst, y_tr, y_tst, pred_prob=False):
    logreg = LogisticRegression()
    logreg.fit(X_tr, y_tr)
    if not pred_prob:
        y_pred = logreg.predict(X_tst)
        acc = accuracy_score(y_tst, y_pred)
        return acc, y_pred
    else:
        y_prob = logreg.predict_proba(X_tst)[:,1]
        roc = roc_auc_score(y_tst, y_prob)
        return roc, y_prob

def print_false_positives(X_test, y_test, y_guess):
    for subj in X_test[y_test < y_pred]:
        print subj, '\n'

def print_false_negatives(X_test, y_test, y_guess):
    for subj in X_test[y_test > y_pred]:
        print subj, '\n'


    
####################################################
####################################################
# |                                              | #
# |         MODELS (STOP WORDS INCLUDED)         | #
# |                                              | #
####################################################
####################################################
    

#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      NAIVE BAYES      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~

# Using only the text of the email
X_train, X_test, y_train, y_test = get_model_data(df, 'text', 123)
train_dtm, test_dtm = transform_data_to_dtm(X_train, X_test, remove_stop_words=False)
acc, y_pred = run_multinomial_nb(train_dtm, test_dtm, y_train, y_test)
roc, y_prob = run_multinomial_nb(train_dtm, test_dtm, y_train, y_test, pred_prob=True)
print 'MODEL: Naive Bayes on email text: \n\t accuracy score: \t %.6f \n\t roc_auc_score: \t %.6f' % (acc, roc)
#MODEL: Naive Bayes on email text: 
#	 accuracy score: 	 0.987177 
#	 roc_auc_score: 	      0.997176

# print confusion matrix
print confusion_matrix(y_test, y_pred)
#[[4509   63]
# [  65 5345]]

# change threshold
y_pred2 = np.where(y_prob > 0.99, 1, 0)
print confusion_matrix(y_test, y_pred2)

# look at false negatives
print_false_negatives(X_test, y_test, y_pred2)
    
# look at false positives:
print_false_positives(X_test, y_test, y_pred2)


### with 150512 data without removing stopwords
# ~~> accuracy score: 0.986259
# ~~> roc_auc_score: 0.99747

### with 150516 data without removing stopwords
# ~~> accuracy score: 0.985752
# ~~> roc_auc_score: 0.997402

# using only the subject line of the email
X_train, X_test, y_train, y_test = get_model_data(df, 'subject', 123)
train_dtm, test_dtm = transform_data_to_dtm(X_train, X_test, remove_stop_words=False)
acc, y_pred = run_multinomial_nb(train_dtm, test_dtm, y_train, y_test)
roc, y_prob = run_multinomial_nb(train_dtm, test_dtm, y_train, y_test, pred_prob=True)
print 'MODEL: Naive Bayes on email Subject: \n\t accuracy score: \t %.6f \n\t roc_auc_score: \t %.6f' % (acc, roc)
#MODEL: Naive Bayes on email Subject: 
#	 accuracy score: 	 0.932779 
#	 roc_auc_score: 	      0.980713

# print confusion matrix
print confusion_matrix(y_test, y_pred)
#[[4259  313]
# [ 358 5052]]
print_false_negatives(X_test, y_test, y_pred)
print_false_positives(X_test, y_test, y_pred)

# ~~> take away: not enough information to use Subject alone
# ~~> need to engineer more features for this model:
# ~~>   . ratio of upper to lower case characters
# ~~>   . number of special characters
# ~~>   . length of subject text 



#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      LOGISTIC REGRESSION      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_train, X_test, y_train, y_test = get_model_data(df, 'text', 1)
train_dtm, test_dtm = transform_data_to_dtm(X_train, X_test, remove_stop_words=False)
acc, y_pred = run_log_reg(train_dtm, test_dtm, y_train, y_test)
roc, y_prob = run_log_reg(train_dtm, test_dtm, y_train, y_test, pred_prob=True)
print 'MODEL: Logistic Regression on email text: \n\t accuracy score: \t %.6f \n\t roc_auc_score: \t %.6f' % (acc, roc)
#MODEL: Logistic Regression on email text: 
#	 accuracy score: 	 0.985774 
#	 roc_auc_score: 	      0.997331

print confusion_matrix(y_test, y_pred) 
#[[4563  110]
# [  32 5277]]
print_false_negatives(X_test, y_test, y_pred)
print_false_positives(X_test, y_test, y_pred)

X_train, X_test, y_train, y_test = get_model_data(df, 'subject', 1)
train_dtm, test_dtm = transform_data_to_dtm(X_train, X_test, remove_stop_words=False)
acc, y_pred = run_log_reg(train_dtm, test_dtm, y_train, y_test)
roc, y_prob = run_log_reg(train_dtm, test_dtm, y_train, y_test, pred_prob=True)
print 'MODEL: Logistic Regression on email text: \n\t accuracy score: \t %.6f \n\t roc_auc_score: \t %.6f' % (acc, roc)
#MODEL: Logistic Regression on email text: 
#	 accuracy score: 	 0.931877 
#	 roc_auc_score: 	      0.981604

print_false_negatives(X_test, y_test, y_pred)
print_false_positives(X_test, y_test, y_pred)




####################################################
####################################################
# |                                              | #
# |         MODELS (STOP WORDS REMOVED )         | #
# |                                              | #
####################################################
####################################################


#  ~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      NAIVE BAYES      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~

# using only the text of the email
X_train, X_test, y_train, y_test = get_model_data(df, 'text', 123)
train_dtm, test_dtm = transform_data_to_dtm(X_train, X_test, remove_stop_words=True)
acc, y_pred = run_multinomial_nb(train_dtm, test_dtm, y_train, y_test)
roc, y_prob = run_multinomial_nb(train_dtm, test_dtm, y_train, y_test, pred_prob=True)
print 'MODEL: Naive Bayes on email text: \n\t accuracy score: \t %.6f \n\t roc_auc_score: \t %.6f' % (acc, roc)
#MODEL: MODEL: Naive Bayes on email text: 
#	 accuracy score: 	 0.987177 
#	 roc_auc_score: 	      0.997020

# print confusion matrix
print confusion_matrix(y_test, y_pred)
#[[4505   67]
# [  61 5349]]
print_false_negatives(X_test, y_test, y_pred)
print_false_positives(X_test, y_test, y_pred)

'''False posivites

An old scam has surfaced recently with renewed vigor The Nigerian-419 fraud 
letter so called because it violates section of Nigerian law is sent in many 
variations by surface and airmail as well as by fax and email Generally the form 
it takes is to ask the unsuspecting victim to provide their bank account 
information in return for a promise to deposit a very large sum of money ...

URGENT NOTIFICATION SI Servers UNAVAILABLE 

CONGRATULATIONS

just in Your application has been pre approved on Wednesday December and your 
mtg process is ready for rates starting at Point Click fill-it out and your 
done http y Yours truly Gale Beatty http 

'''


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  |      LOGISTIC REGRESSION      |
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_train, X_test, y_train, y_test = get_model_data(df, 'text', 1)
train_dtm, test_dtm = transform_data_to_dtm(X_train, X_test, remove_stop_words=True)
acc, y_pred = run_log_reg(train_dtm, test_dtm, y_train, y_test)
roc, y_prob = run_log_reg(train_dtm, test_dtm, y_train, y_test, pred_prob=True)
print 'MODEL: Logistic Regression on email text: \n\t accuracy score: \t %.6f \n\t roc_auc_score: \t %.6f' % (acc, roc)
#MODEL: Logistic Regression on email text: 
#	 accuracy score: 	 0.984472 
#	 roc_auc_score: 	      0.997101

# print confusion matrix
print confusion_matrix(y_test, y_pred)
#[[4549  124]
# [  31 5278]]
print_false_negatives(X_test, y_test, y_pred) # appear to be very long emails ...
print_false_positives(X_test, y_test, y_pred) # appear to be very short emails (or spam-like)

# TAKEAWAYS
# ~~> Model catches too many ham emails (i.e., producs too many false positives)
# ~~> 
# ~~> 

# Examples of false positives
'''
Sally Thought this was good Cindy Hey here is another wonderful group of facts 
about women I do n't normally like to pass these on but it has too much 
importance Even if your a skeptic read to the bottom Did you know that it 
Beautiful Women Month Well it is and that means you ...

Thought you might enjoy this An old man a boy and a donkey were going to town 
The boy rode on the donkey and the old man walked As they went along they passed 
some people who remarked it was a shame the old man was walking and the boy was
 riding The man and boy thought maybe the critics were right so they changed 
 positions Later they passed some people that remarked What a shame he makes 
 that little boy walk They then decided they both would walk Soon they passed 
 some more people who thought they were stupid to walk when they had a decent 
 donkey to ride So they both rode the donkey Now they passed some people that 
 shamed them by saying how awful to put such a load on a poor donkey The boy 
 and man said they were probably right so they decided to carry the donkey 
 As they crossed the bridge they lost their grip on the animal and he fell 
 into the river and drowned The moral of the story If you try to please everyone 
 you might as well kiss your ass good-bye 
 
 DELTA FAN FARES FOR FEBRUARY FEBRUARY Hello Mr Farmer Welcome to this week 
 version of Delta Fan Fares For the uninitiated it an incredible e-mail program
 for Delta customers who want to get away from it all and enjoy events and 
 activities in cities across the country Before you start packing remember 
 that you need your SkyMiles number...

'''    

# create our own y_pred using y_prob
print '|Probability threshold \t| Accuracy score \t| False positives \t| False negatives \t| Total misses \t|'
print '|----------- |-------------------|----------|---------|-----------------|'
prob = 0.50
while prob < 1:    
    y_pred = np.where(y_prob > prob, 1, 0)
    acc = accuracy_score(y_test, y_pred)
    conmat = confusion_matrix(y_test, y_pred)
    print '|%.2f \t| %.4f \t| %d \t| %d \t|  %d \t|' % (prob, acc, conmat[0][1], conmat[1][0], conmat[0][1] + conmat[1][0])
    prob += 0.05

# relative performance gain by setting a probability threshold at 0.55 or 0.60
#Prob: 0.50 	| Accuracy: 0.9845 	| FP: 124 	| FN: 31 	| TotalMiss: 155 	|
#Prob: 0.55 	| Accuracy: 0.9853 	| FP: 113 	| FN: 34 	| TotalMiss: 147 	|
#Prob: 0.60 	| Accuracy: 0.9852 	| FP: 106 	| FN: 42 	| TotalMiss: 148 	|
#Prob: 0.65 	| Accuracy: 0.9861 	| FP: 93 	| FN: 46 	| TotalMiss: 139 	|
#Prob: 0.70 	| Accuracy: 0.9871 	| FP: 76 	| FN: 53 	| TotalMiss: 129 	|
#Prob: 0.75 	| Accuracy: 0.9869 	| FP: 69 	| FN: 62 	| TotalMiss: 131 	|
#Prob: 0.80 	| Accuracy: 0.9865 	| FP: 59 	| FN: 76 	| TotalMiss: 135 	|
#Prob: 0.85 	| Accuracy: 0.9680 	| FP: 39 	| FN: 280 	| TotalMiss: 319 	|
#Prob: 0.90 	| Accuracy: 0.9626 	| FP: 26 	| FN: 347 	| TotalMiss: 373 	|
#Prob: 0.95 	| Accuracy: 0.9474 	| FP: 19 	| FN: 506 	| TotalMiss: 525 	|




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# |        BAGGING NAIVE BAYES         | 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


np.random.seed(123)

def get_bootstrap_sample(data):
    samp = np.random.choice(a=data.shape[0], size=data.shape[0]*0.50, replace=False)
    oob = list(set(range(data.shape[0])) - set(samp))   
    boot = data.iloc[samp, :]
    oob  = data.iloc[oob, :]
    X_boot, y_boot = boot.text, boot.spam
    X_oob, y_oob = oob.text, oob.spam
    return X_boot, X_oob, y_boot, y_oob

# bagging
NUM_BOOT = 2
predictions = [0]*len(y_test)
for ii in xrange(NUM_BOOT):
    X_train, X_test, y_train, y_test = get_bootstrap_sample(df)
    
    # fit and transform X_train, and transform X_test
    vect = CountVectorizer(stop_words='english')
    train_dtm = vect.fit_transform(X_train)
    test_dtm = vect.transform(X_test)
    
    # run Multinomial Naive Bayes
    mn_nb = MultinomialNB()
    mn_nb.fit(train_dtm, y_train)
    
    # make predictions
    y_pred = mn_nb.predict(test_dtm)
    acc = accuracy_score(y_test, y_pred) 
    print "print %d: %.5f" % (ii, acc)     
    #confusion_matrix(y_test, y_pred)
    predictions = predictions + y_pred









#  HOW DO I TUNE NB AWAY FROM FALSE POSITIVES?
# ~~> LogReg: predict proba, change cut off values (first, see if FP and FN are close to 50%?)
# ~~> NB: ???
