# Mechanically Separated Ham and Spam (DRAFT)
###May 18, 2015

****

###Problem and hypothesis

The problem is simple. The world is full of awful spammers who flood people's email boxes with nonsense at best, and with malious, identity stealing malware at worst. The spammers must be stopped. To do this we need to reduce the number of spam emails that people see, and spam filtering is the way to do that.

My personal motivation for this project is twofold:

1. To understand better how SPAM filters work in the real world
2. To extract information from unstructured text and use that information to make predictions - specifically, to use email text and other features to prediction whether an email is ham or spam.


My hypothesis is that spam emails will contain more words and phrases about money, "great deals", sex, and sender appeals for help.

###Description of data

The data that will be used for this project comes from two main sources. A subset of the [Enron email corpus](https://www.cs.cmu.edu/~./enron/) which entered into the public domain during the Federal Energy Regulatory Commission's investigation of Enron after Enron's [spectacular collapse](http://www.salon.com/2003/10/14/enron_22/). Specifically, the subset used for this project includes the email of six executives. The spam messages come from the SpamAssassin corpus, the Honeypot Project, the personal spam archives of researchers Bruce Guenter and Georgios Paliouras. A copy of the data can be downloaded at http://www.aueb.gr/users/ion/data/enron-spam/.

The raw data contain 52,075 email files.

###Preprocessing the data

Taking the data from raw form to a form that could be more easily analyzed was the most difficult part of this project. In order to learn more about Natural Language Processing, I chose to use the email data in its raw form. This included emails that had different character encodings, emails with and without html elements, and a variety of different underlying structures. Decisions made in the part of the process will inevitably influence the modeling phrase, but, given the size of the sample, the results may not be terribly large.

In order to clean the data, I first had to take an inventory of all the data that I had (see [ raw\_data\_inventory.py](https://github.com/jw-ml/dat5\_spam-filter/blob/master/code/raw\_data\_inventory.py)). After taking the data inventory, I try to extract information from each email and to give more structure to the data (see [raw\_to\_clean\_data.py](https://github.com/jw-ml/dat5_spam-filter/blob/master/code/raw_to_clean_data.py)). Specifically, with varying degrees of success, I follow the following steps:

1. Use python's [email](https://docs.python.org/2/library/email.html) module to separate the email's body (text) from the email's metadata (headers, such as 'To', 'From', 'Subject', and 'Encoding')
2. Use [beautifulsoup](http://www.crummy.com/software/BeautifulSoup/) to strip out all html tags that might be included in the email
3. Remove any text that is included as part of a "reply" or "forward" (_note: Leaving this data in could have a large impact on the model's predictions. I indent to run the models again with replies and forwards included before the final write-up_)

At the end of this cleaning, I construct a dataframe with four columns:
1. 'spam': whether the email is ham or spam
2. 'timestamp': the time the email was sent (or received - it is not always clear)
3. 'subject': the email's subject line
4. 'text': the body of the email

The above turned out to be surprisingly difficult (for me). The emails came in lots of formats, and several of those formats did not play well with the email or beautifulsoup modules. A major area of improvement for this project is in the data cleaning stage and better exception/error handling.

Lastly, I use the natural language toolkit (nlkt) and scikit-learn's CountVectorizer to further clean text elements and to create word count vectors (see [preprocess\_and\_model.py](https://github.com/jw-ml/dat5_spam-filter/blob/master/code/preprocess_and_model.py)).

###Data exploration and visualization

#_INSERT WORDCOUNTS_
#_INSERT WORD CLOUD_

###Features

Currently, my model is built around only a few features:
1. the document term matrix associated with the email text
2. the document term matrix associtaed with the email subject

Over the course of time, I intend to create the following features:
1. the length of text
2. the length of subject
3. ratio of upper to lower case letters
4. number of special (i.e., non-alphanumeric) characters
5. number of links included in email (not sure how to do this one yet...)
6. features related to the timestamp (not sure how to do this one yet...)
7. incorporating n-grams of size two into the document term matrices above


###Model selection and results

The models I have selected to apply to the document term matrices above are Multinomial Naive Bayes and Logistic Regression. Currently, they both perform well....

###Main challenges

###Areas for further work

1. Creation of new features, such as:
  * count number of links in email
  * ratio of uppercase to lowercase characters
  * number of non-alpahnumeric characters
  * length of email, subject, etc.
  * timestamp (hour, day of the week, etc.)
  * including n_grams
2. Creation of ensemble method to combine of naive bayes on email body and subject, and logistic regression (or others) on derived features listed above.

###Conclusions
