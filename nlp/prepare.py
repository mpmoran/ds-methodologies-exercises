# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # NLP - Preparation Exercise

# +
import re
import unicodedata
from functools import reduce, partial
from copy import deepcopy

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


# +
# right to left
def compose(*fns):
    return partial(reduce, lambda x, f: f(x), reversed(fns))


# applies in the order supplied
def pipe(v, *fns):
    return reduce(lambda x, f: f(x), fns, v)


def map_exhaust(func, *iters):
    for args in zip(*iters):
        func(*args)


# +
def normalize_text(text):
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def remove_chars(text):
    return re.sub(r"[^A-Za-z0-9\s']", "", text)


def basic_clean(text):
    return pipe(text, str.lower, normalize_text, remove_chars)


# -


def tokenize(text):
    tokenizer = ToktokTokenizer()
    return tokenizer.tokenize(text, return_str=True)


def stem(text):
    ps = nltk.porter.PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])


def lemmatize(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(lemmas)


def remove_stopwords(text, include=[], exclude=[]):
    stopword_list = stopwords.words("english")

    map_exhaust(stopword_list.remove, exclude)
    map_exhaust(stopword_list.append, include)
    
    removed = " ".join([w for w in text.split() if w not in stopword_list])

    
#     print("Removed", len(text.split()) - len(removed.split()), "words")
    return removed


def prep_article(article):
    copy = deepcopy(article)

    copy["clean"] = pipe(copy["original"], basic_clean, tokenize, remove_stopwords)

    copy["stemmed"] = stem(copy["clean"])

    copy["lemmatized"] = lemmatize(copy["clean"])

    return copy


def prepare_article_data(articles):
    return (prep_article(a) for a in articles)
