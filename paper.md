# Mechanically Separated Ham and Spam (DRAFT)
###May 18, 2015

****

###Problem and hypothesis

The problem is simple. The world is full of awful spammers who flood people's email boxes with nonsense at best, and with malious, identity stealing malware at worst. The spammers must be stopped. To do this, we need to reduce the number of spam emails that people see, and spam filtering is the way to do that.

My personal motivation for this project is twofold:

1. To understand better how SPAM filters work in the real world
2. To extract information from unstructured text and use that information to make predictions - specifically, to use email text and other features to prediction whether an email is ham or spam.


My hypothesis is that spam emails will contain more words and phrases about money, "great deals", sex, and sender appeals for help.

###Description of data

The data that will be used for this project comes from two main sources. A subset of the [Enron email corpus](https://www.cs.cmu.edu/~./enron/) which entered into the public domain during the Federal Energy Regulatory Commission's investigation of Enron after Enron's [spectacular collapse](http://www.salon.com/2003/10/14/enron_22/). Specifically, the subset used for this project includes the email of six executives. The spam messages come from the SpamAssassin corpus, the Honeypot Project, the personal spam archives of researchers Bruce Guenter and Georgios Paliouras. A copy of the data can be downloaded at http://www.aueb.gr/users/ion/data/enron-spam/.

The raw data contain 52,075 email files.

###Preprocessing the data

Taking the data from raw form to a form that could be more easily analyzed was the most difficult part of this project. In order to learn more about Natural Language Processing, I chose to use the email data in its raw form instead of the preprocessed form which is also available online. The raw data included emails that had different character encodings, emails with and without html elements, and a variety of different underlying structures. Decisions made in the part of the process will inevitably influence the modeling phrase, but, given the size of the sample, the effects may not be terribly large.

In order to clean the data, I took an inventory of all the data that I had (see [ raw\_data\_inventory.py](https://github.com/jw-ml/dat5\_spam-filter/blob/master/code/raw\_data\_inventory.py)). After taking the data inventory, I attempt to extract information from each email and to give more structure to the data (see [raw\_to\_clean\_data.py](https://github.com/jw-ml/dat5_spam-filter/blob/master/code/raw_to_clean_data.py)). Specifically, I followed the following steps (with varying degrees of success):

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

* _need to insert word counts_
* _try to create a word cloud_

###Features

Currently, my model is built around only a few features:
1. the document term matrix associated with the email text
2. the document term matrix associtaed with the email subject

Over the next week, I intend to create the following features:
1. the length of text
2. the length of subject
3. ratio of upper to lower case letters
4. number of special (i.e., non-alphanumeric) characters
5. number of links included in email (not sure how to do this one yet...)
6. features related to the timestamp (not sure how to do this one yet...)
7. incorporating n-grams of size two into the document term matrices above


###Model selection and results

The models I have selected to apply to the document term matrices above are Multinomial Naive Bayes and Logistic Regression. To fit these models, I use train-test-split on the data, fit the model, make predictions, and then calculate the accuracy score and roc_auc_score for each model. Here is a summary the results of four models:

#####Summary of initial models and results (_features = email text with replies/forwards included_)
| Model                                             | Accuracy Score | ROC-AUC-Score  |
| -------------                                     |:-------------: |:-------: |
| Naive Bayes (without removing stop words)         | 0.987177       | 0.997176 |
| Naive Bayes (with stop words removed)             | 0.987177       | 0.987177 |
| Logistic Regression (without removing stop words) | 0.985774       | 0.997331 |
| Logistic Regression (with stop words removed)     | 0.984472       | 0.997101 |

Oddly, removing the stop words actually resulted in _slightly_ less accurate precitions and roc-auc-scores for the logistic regressions. 

The other feature I (currently) have is the email Subject line. Running this feature through the same set of models, I generated the following results. Because email Subjects are much shorter on average, I did not bother removing stop words (although, that may be a useful addition).

#####Summary of initial models and results (_features = email subject line_)
| Model                                             | Accuracy Score | ROC-AUC-Score  |
| -------------                                     |:-------------: |:-------: |
| Naive Bayes (without removing stop words)         | 0.932779       | 0.980713 |
| Logistic Regression (without removing stop words) | 0.931877       | 0.981604 |

Again, Naive Bayes performs slightly better than Logistic Regression, but does not perform nearly as well as the running the same models on the body of the email.

Next, I investigated out the models were missing their predictions. In general, people believe that a False Positive (FP) in a spam filter is much worse than a False Negative (FN). That is, people do not mind _some_ spam making it into their inbox, but they do not like ham going to their spam box. Here are the initial results:

#####Confusion matrix (_features = email text with replies/forwards included_)
| Model                                             | True negatives | False positives | False negatives | True positives |
| -------------                                     |:---: |:--: |:--: |:--: |
| Naive Bayes (without removing stop words)         | 4509 | 63  |65   |5345 |
| Naive Bayes (with stop words removed)             | 4505 | 67  |61   |5349 |
| Logistic Regression (without removing stop words) | 4563 | 110 |32   |5277 |
| Logistic Regression (with stop words removed)     | 4549 | 124 |31   |5278 |

From the confusion matrix, it is easy to see that Naive Bayes is outperforming Logistic Regression. Even when we tune the logistic regression to have a larger classification threshold, we cannot beat Naive Bayes.

So, what types of emails is Naive Bayes (without removing stop words) missing? Here are few examples

#####False positives
> Dear Ms Kitchen You are a strong contender to be on Fortune s list of the Most Powerful Women in Business this year Please take a moment to make our Powerful Women issue as engaging and fun for our magazine readers and website users as it has been for us By answering the following questions you can help us to get a better sense of you and your busy life...

> Energy Info Source is privileged to make available to you a free sample issue of its Bi-weekly Transmission Update Report attached This report contains the latest ISO RTO Utility and Merchant Transmission news ...

> For my fellow travelers who will be going to Spain Italy Seattle and Pennsylvania Interesting travel web site ...

#####False negatives
> Doctors Use This Too Stay hard for straight incrrease si.ize and staa_mina with one piilll En+terr here http vsale I would come home late in the night and would get out early in the morning ...

> Final Notice Hi I sent you an email last week and need to confirm everything now Please read info below and let me know if you have any questions We are accepting your mortgage application ...

> Good day I tried to call your three time but your phone is not available I think you did a mistake during filling the form Anyway your mortgagge request was appproved with please reenter your info here and we will start ASAP http Thank you Luke MCPKVL 

For next steps on the modeling, I intend to create the features listed above, and to create an ensemble method combining various models. Hopefully, I will find some feature that will help me tune my model such that it predicts fewer false positives.


###Main challenges

The main challenge with this project was processing the email data. In fact, that battle is far from over ...

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
