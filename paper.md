# Mechanically Separated Ham and Spam (DRAFT)
###May 18, 2015

****

###Problem and hypothesis

The problem is simple. The world is full of spammers who flood people's email boxes with nonsense at best, and with identity stealing malware at worst. The spammers must be stopped. To do this, we need to reduce the number of spam emails that people see, and spam filtering is the way to do that.

My personal motivation for this project is twofold:

1. To understand better how SPAM filters work in the real world
2. To extract information from unstructured text and use that information to make predictions - specifically, to use email text and other features to predict whether an email is ham or spam.


My hypothesis is that spam emails will contain more words and phrases about money, "great deals", sex, and sender appeals for help.

###Description of data

The data that will be used for this project comes from two main sources. A subset of the [Enron email corpus](https://www.cs.cmu.edu/~./enron/) which entered into the public domain during the Federal Energy Regulatory Commission's investigation of Enron after Enron's [spectacular collapse](http://www.salon.com/2003/10/14/enron_22/). Specifically, the subset used for this project includes the email of six executives. The spam messages come from the SpamAssassin corpus, the Honeypot Project, the personal spam archives of researchers Bruce Guenter and Georgios Paliouras. A copy of the data can be downloaded at http://www.aueb.gr/users/ion/data/enron-spam/.

The raw data contain 52,075 _raw email_ files containing 19,088 ham emails, and 32,988 spam emails. After cleaning the email and preprocessing the data for analysis, I also drop a number of emails.

#####Email sample sizes
| Data                                              | Ham     | Spam   | Total (N)  | 
| -------------                                     |:-------:|:------:|:----------:|
| Raw data                                          | 19,088  | 32,988 | 52,076     |
| "clean" data                                      | 18,962  | 22,006 | 40,968     |
| preprocessed data (with null values dropped)      | 18,657  | 21,270 | 39,927     |


For the above numbers and the results included in this paper, I use a sample that includes email replies and forwarded emails. One could rerun the analysis without including replies and forwards, but that would lead to a smaller sample size with slightly less predictive power.

Also, at first I thought my results would be skewed because I was using only email from a small subset of Enron employees. However, spam filters are inherently individualized and I now think my results may be skewed in the other direction because I am comingling emails from several people. Another potential project may be to only run these models on emails sent from one individual.

###Cleaning and preprocessing the data

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

The above turned out to be surprisingly difficult (for me). The emails came in lots of formats, and several of those formats did not play well with the email or beautifulsoup modules. A major area of improvement for this project is in the data cleaning stage and better exception/error handling. Currently, I have to drop several emails from the data because I still have not figured out how to deal with those emails (hence, the decrease in N seen above).

Lastly, I use the natural language toolkit (nlkt) and scikit-learn's CountVectorizer to further clean text elements and to create word count vectors (see [preprocess\_and\_model.py](https://github.com/jw-ml/dat5_spam-filter/blob/master/code/preprocess_and_model.py)).


###Features

Currently, my model is built around only a few features of the possible features that can be extracted from this data:

1. the document term matrix associated with the email text (p = 181,870)
2. the document term matrix associtaed with the email subject (p = 19,435)

Other features to investigate could include:

1. the length of text
2. the length of subject
3. ratio of upper to lower case letters
4. number of special (i.e., non-alphanumeric) characters
5. number of links included in email (not sure how to do this one yet...)
6. features related to the timestamp (not sure how to do this one yet...)
7. incorporating n-grams of size two into the document term matrices above


###Model selection and results

The models I have selected to apply to the document term matrices above are Multinomial Naive Bayes and Logistic Regression. To fit these models, I use train-test-split on the data, fit the model, make predictions, and then calculate the accuracy score and roc_auc_score for each model. To create a baseline estimate, I ran naive bayes on the emails in a completely unprocessed form. I simply decoded them into unicode, and loaded them into a dataframe. For the other models, I tried different approaches to preprocessing the data. Here is a summary the results.

#####Summary of baseline model and results(_features = all information contained in raw data emails_)
| Model                                             | Accuracy Score | ROC-AUC-Score  |
| -------------                                     |:-------------: |:-------: |
| Naive Bayes (without removing stop words)         | 0.998464       | 0.998671 |


#####Summary of initial models and results with preprocessing (_features = email text with replies/forwards included_)
| Model                                             | Accuracy Score | ROC-AUC-Score  |
| -------------                                     |:-------------: |:-------: |
| Naive Bayes (without removing stop words)         | 0.987177       | 0.997176 |
| Naive Bayes (with stop words removed)             | 0.987177       | 0.987177 |
| Logistic Regression (without removing stop words) | 0.985774       | 0.997331 |
| Logistic Regression (with stop words removed)     | 0.984472       | 0.997101 |

Oddly, contrary to my initial beliefs, removing any sort of information from the raw data generally resulted in less accurate precitions and roc-auc-scores. The best performing model on my sample was the baseline case in which I did zero preprocessing of the data.

The other feature I (currently) have is the email Subject line. Running this feature through the same set of models, I generated the following results. Because email Subjects are much shorter on average, I did not bother removing stop words.

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
| Naive Bayes (baseline model)                      | 4,769 | 4  |16   |8,230 |
| Naive Bayes (without removing stop words)         | 4,509 | 63  |65   |5,345 |
| Naive Bayes (with stop words removed)             | 4,505 | 67  |61   |5,349 |
| Logistic Regression (without removing stop words) | 4,563 | 110 |32   |5,277 |
| Logistic Regression (with stop words removed)     | 4,549 | 124 |31   |5,278 |

From the confusion matrix, it is easy to see that Naive Bayes is outperforming Logistic Regression. Even when we tune the logistic regression to have a larger classification threshold, we cannot beat Naive Bayes.

So, what types of emails is Naive Bayes (without removing stop words) missing? Here are some examples.

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

The main challenge with this project was processing the email data. However, as the results above indicate, adding more structure to the data did not improve the predictive ability of the models. In fact, the opposite was true. However, this may be due to a large number of spam emails having html tags included, and all of the Enron emails containing at least one of six email addresses (email addresses that were not contained in the spam sample).

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

Naive Bayes is a very powerful tool. Even initial runs of the models were returning accuracy rates greater than 98%. Further, there also a large number of potential features that can be created from text documents.
