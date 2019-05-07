#!/usr/bin/env python
# coding: utf-8

# # NLP - Preparation Exercise

# In[ ]:


import re
import unicodedata
from functools import reduce, partial
from copy import deepcopy
from pprint import pprint

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd


# In[ ]:


# right to left
def compose(*fns):
    return partial(reduce, lambda x, f: f(x), reversed(fns))


# applies in the order supplied
def pipe(v, *fns):
    return reduce(lambda x, f: f(x), fns, v)


def map_exhaust(func, *iters):
    for args in zip(*iters):
        func(*args)


# In[ ]:


def normalize_text(text):
    return unicodedata.normalize('NFKD', text)                      .encode('ascii', 'ignore')                      .decode('utf-8', 'ignore')


def remove_chars(text):
    return re.sub(r"[^A-Za-z0-9\s']", "", text)


def basic_clean(text):
    return pipe(text, str.lower, normalize_text, remove_chars)


# In[ ]:


def tokenize(text):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(text, return_str=True)


# In[ ]:


def stem(text):
    ps = nltk.porter.PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])


# In[ ]:


def lemmatize(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(lemmas)


# In[ ]:


def remove_stopwords(text, include=[], exclude=[]):
    stopword_list = stopwords.words('english')
    
    map_exhaust(stopword_list.remove, exclude)
    map_exhaust(stopword_list.append, include)
    
    
    return " ".join([w for w in text.split() if w not in stopword_list])


# In[ ]:


def prep_article(article):
    copy = deepcopy(article)
    
    clean_toked = pipe(copy["original"], basic_clean, tokenize)
    
    copy["stemmed"] = stem(clean_toked)
    copy["clean_stemmed"] = remove_stopwords(copy["stemmed"])
    
    copy["lemmatized"] = lemmatize(clean_toked)
    copy["clean_lemmatized"] = remove_stopwords(copy["lemmatized"])
    
    return copy


# In[ ]:


def prepare_article_data(articles):
    return (prep_article(a) for a in articles)

