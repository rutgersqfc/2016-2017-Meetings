
# coding: utf-8

# # Introduction
# 
# This presentation aims to prepare club members with some basic tools and knowledge to succeed in the upcoming Quant Quest challenge.
# 
# ## Machine Learning Pipeline
# 1. Obtain data
#   * Either from scraping, downloading, or other means.
# 2. Preprocess data
#   * Remove unwanted data.
#   * Filter out noise.
#   * Patitioning data into *training set*, *validation set*, *test set*
#   * Scale, shift, and normalize.
# 3. Find a good representation
#   * The purpose of this step is to find a more representative representation of the data. 
#   * In NLP, a good representation can be *word count*, or *tf-idf*.
#   * Dimensionality reduction.
# 4. Training the classifier/regressor
#   * People often [k-fold cross-validation](https://www.cs.cmu.edu/~schneide/tut5/node42.html).
#   * *training* is done using gradient descent.
#   * Hyper-parameters tuning.
# 5. Testing
#   * Accuracy, false-positive, false-negative, f-1 score, etc.

# 
# 
# ## Some Tools
# This section details some Python libraries that might be helpful
# 1. Numerical analysis
#   * [numpy](http://www.numpy.org/) - Linear algebra, matrix and vector manipulation
#   * [pandas](http://pandas.pydata.org/) - Data anaysis, data manipulation
# 2. Machine learning
#   * [scikit-learn](http://scikit-learn.org/stable/) - General machine learning. Supports basic/advance level algorithms, but only run on CPU.
#   * [theano](http://deeplearning.net/software/theano/) - Deep learning framework.
#   * [tensorflow](https://www.tensorflow.org/) - Another deep learning framework.
# 3. Natural language processing
#   * [nltk](http://www.nltk.org/) - General NLP
#   * [gensim](https://radimrehurek.com/gensim/) - Topic modeling
# 4. Utilities
#   * [beautiful soup](https://www.crummy.com/software/BeautifulSoup/) - Utility for working with text
#   * [urllib](https://docs.python.org/2/library/urllib2.html) - Dealing with url, lightweight scraping.
#   * [wikipedia](https://wikipedia.readthedocs.io/en/latest/quickstart.html) - Scraping from wikipedia
#   
# ## Download
# You can get most of these libraries from the [Anaconda distribution](https://www.continuum.io/downloads) or from the links above.

# ## Obtain data
# This section will introduce basic tools to download text corpus from wikipedia articles. We will download the content of all 500 articles of 500 companies in the S&P 500.

# In[ ]:

import urllib2
import string
import time
import os
from bs4 import BeautifulSoup, NavigableString
import wikipedia as wk


# In[ ]:

def initOpener():
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    return opener


# The function below output a dictionary whose keys are *stock tickers* and values are article *URLs*. These *URLs* are then used for scraping.

# In[ ]:

def getSP500Dictionary():
    stockTickerUrl = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    usableStockTickerURL = initOpener().open(stockTickerUrl).read()

    stockTickerSoup = BeautifulSoup(usableStockTickerURL, 'html.parser')

    stockTickerTable = stockTickerSoup.find('table')

    stockTickerRows = stockTickerTable.find_all('tr')

    SP500companies = {}

    stockBaseURL = 'https://en.wikipedia.org'

    for stockTickerRow in stockTickerRows:
        stockTickerColumns = stockTickerRow.find_all('td')
        counter = 1
        for element in stockTickerColumns:
            # Stock Ticker
            if (counter % 8) == 1:
                stockTicker = element.get_text().strip().encode('utf-8', 'ignore')
                counter = counter + 1
            # Corresponding link to wiki page
            elif (counter % 8 == 2):
                SP500companies[stockTicker] = element.find('a', {'href': True}).get('href')
                counter = counter + 1

    return SP500companies


# The cell bellow uses *wikipedia* package to load the summary paragraph of the wikipedia article of each company.

# In[ ]:

import codecs
import wikipedia as wk
import sys
import json

SP500dict = getSP500Dictionary()
err = []
data = []
comp_name = []
for k, v in SP500dict.iteritems():
    # k: ticker, v: company name
    v_str = str(v)
    pageId = v_str.split('/')[-1]
    pageId = pageId.replace('_',' ')
    try:
        data.append(wk.summary(pageId).encode('utf-8'))
        comp_name.append(pageId.encode('utf-8'))
    except:
        err.append((k,v))
# Dump the data into json file for later use
with open('data.json', 'w') as outfile:
    json.dump((data, comp_name), outfile)


# In[1]:

import json

with open('data.json') as json_data:
    data_ = json.load(json_data)
data = data_[0]
comp_name = data_[1]
# print 2 companies
print data[10]
print '-----'
print data[11]


# ## Preprocessing and Feature Representation
# Vectorize documents to matrix of occurence. While counting, filter out stopwords.

# In[2]:

# Import the method
from sklearn.feature_extraction.text import CountVectorizer
# Initialize the vectorizer with the option of stopword, which will eliminate 
# common words like 'the', 'a', etc.
count_vect = CountVectorizer(stop_words='english')
# fit_transform method applies the vectorizer on the data set
X_train_counts = count_vect.fit_transform(data)
# The resulting matrix is 496 by 7942. Each row is a document (a wikipedia article)
# each column is the occurence of each word.
print X_train_counts.shape


# $tf(t,d)$ is the frequency that term $t$ appears in document $d$.
# 
# $df(d,t)$ is the number of documents that contain term $t$.
# 
# $idf(t)=\log \frac{1+n_d}{1+df(d,t)} + 1$, 
#   * $n_d$ is number of documents
# 
# $tfidf(t,d)=tf(t,d)\times idf(t)$
# 
# In sklearn implementation, the final tf-idf vector is normalized by the L2 norm.
# 
# Tfidf gives a nice numerical representation of the document. From this representation, we can perform numerical analysis technique on the data.

# In[3]:

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer()
X_train_tf = tf_transformer.fit_transform(X_train_counts)
print X_train_tf.shape
print 


# ## Clustering
# K-means cluster your dataset into K centroids. 
# 
# For a set of observation $(x_0, x_1, \dots, x_n)\in \mathbb{R}^d$ (in our case, $n=498$ and $d=7940$), k-means clusters these $n$ observations into $k$ groups $S={S_1, S_2, \dots, S_k}$ such that: 
# 
# $$argmin_S \sum_{i=1}^{k} \sum_{x\in S_i} ||x-\mu_i||^2$$
# 
# Intuitively, we want to minimize the total distance of each point in a cluster to the center $\mu$ of that cluster.
# 
# We start with placing centroids on the data set (there are many schemes to initialize centroids, but we go with random). Then for each data point, we determine which group it belongs to by looking at the Euclidian distance. 
# 
# Next, we iteratively update the center to minimize the sum of distance of all points in that group to the group center. At each iteration, the new centroid is the arithmetic mean of all points in that cluster.

# In[4]:

from sklearn.cluster import KMeans
# Note that n_clusters is number of cluster. This is important for accuracy. Play around with it
classifier = KMeans(n_clusters = 90, n_jobs=-1)
classifier.fit(X_train_tf)


# In[5]:

print (classifier.labels_)


# In[6]:

import numpy as np
print [comp_name[x] for x in np.where(classifier.labels_==30)[0]]
print "____"
print [comp_name[x] for x in np.where(classifier.labels_==35)[0]]


# In[7]:

print comp_name.index('Goldman Sachs Group')
print [comp_name[x] for x in np.where(classifier.labels_==classifier.labels_[118])[0]]


# In[9]:

# print comp_name

