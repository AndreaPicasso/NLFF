
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np



def lemmatizeDictionary(dictionary):
    # Calling our overwritten Count vectorizer
    #newDict = pd.DataFrame(columns = list(dictionary.columns.values))
    porter_stemmer = PorterStemmer()
    cols = list(dictionary.columns.values)

    for i, row in dictionary.iterrows():
        for column in cols:
            dictionary[column].at[i] = porter_stemmer.stem(row[column])

    



def getSentiment(sentence, dictionary):
    #
    porter_stemmer = PorterStemmer()
    tokens = word_tokenize(sentence)

    sentence = [porter_stemmer.stem(x) for x in tokens]
    POSITIVE = set(sentence) & set(dictionary['POSITIVE'].tolist())
    NEGATIVE = set(sentence) & set(dictionary['NEGATIVE'].tolist())
    UNCERTAINTY = set(sentence) & set(dictionary['UNCERTAINTY'].tolist())
    LITIGIOUS = set(sentence) & set(dictionary['LITIGIOUS'].tolist())
    CONSTRAINING = set(sentence) & set(dictionary['CONSTRAINING'].tolist())
    #print(CONSTRAINING)
    totWord = len(POSITIVE) + len(NEGATIVE) + len(UNCERTAINTY) + len(LITIGIOUS) + len(CONSTRAINING)
    if totWord == 0:
        totWord = 1

    return {'NEGATIVE':len(NEGATIVE)/totWord,
    'POSITIVE':len(POSITIVE)/totWord,
    'UNCERTAINTY':len(UNCERTAINTY)/totWord,
    'LITIGIOUS':len(LITIGIOUS)/totWord,
    'CONSTRAINING':len(CONSTRAINING)/totWord
    }





   
