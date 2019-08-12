from __future__ import division
import pandas as pd
import json
import ast
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import sent_tokenize, word_tokenize, pos_tag
import re

max_value=2.8373539548245974
min_value=-2.258099953637726
emoticons_str = r"""
(?:
    [:=;] # Eyes
    [oO\-]? # Nose (optional)
    [D\)\]\(\]/\\OpP] # Mouth
)"""
regex_str = [
    emoticons_str,
    r'(?:[\.]{2,10})',  # suspension points
    r'(?:[Ss]+[&]+[Pp]+)',
    r'\\(?:[a-z0-9]{3,3}[\\])+[a-z0-9]{3,3}',  # unicode
    r"(?:\$+[\w_]*[.]?[\d]*)",  # jargons
    # r'\\n\\n',#enter and next line
    r'\\n',
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-.]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs
    r'(?:(?:\.?\d+,?-?%?\$?)+(?:\.?\/?\d+\/?\d*)?(?:[cb])?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

#string='hi, houses andrea bullish bearish #hello'
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE)
lemmatizer = WordNetLemmatizer()
def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=True):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def estractSentimentFil(string):

    with open("list_OL_FINANCE.txt", "r") as myfile:
        data = myfile.read()
    data = ast.literal_eval(data)
    sentimentFil = json.loads(json.dumps(data))
    string = preprocess(string)
    sentiment = 0
    count = 0
    for el in string:
        lemma = lemmatizer.lemmatize(el)
        # print(lemma)
        try:
            sentiment += sentimentFil[lemma]
            count += 1
        except:
            f=0  
    
    if (count == 0):
        result = 100
    else:
        result = sentiment / count
        result=(2*(result-min_value)/(max_value-min_value))-1
        #print(result)
    return result