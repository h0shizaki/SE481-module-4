import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import util
from nltk.corpus import stopwords
import pandas as pd

def create_stem_cache(cleaned_description):
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)
    return stem_cache

def create_custom_preprocessor(stop_dict, stem_cache):
    def custom_preprocessor(s) :
        ps = PorterStemmer()
        s = re.sub(r'[^A-Za-z]', ' ', s)
        s = re.sub(r'\s+', ' ' , s)
        s = word_tokenize(s)
        s = list(set(s) - stop_dict)
        s = [word for word in s if len(word) > 2]
        s = [stem_cache[w] if w in stem_cache else ps.stem(w) for w in s]
        s = ' '.join(s)
        return s
    return custom_preprocessor

def preprocess(s, stop_dict , stem_cache) :
    ps = PorterStemmer()
    s = re.sub(r'[^A-Za-z]', ' ', s)
    s = re.sub(r'\s+', ' ' , s)
    s = word_tokenize(s)
    s = list(set(s) - stop_dict)
    s = [word for word in s if len(word) > 2]
    s = [stem_cache[w] if w in stem_cache else ps.stem(w) for w in s]
    s = ' '.join(s)
    return s