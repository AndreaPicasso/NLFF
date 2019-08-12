
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
    SUPERFLUOUS = set(sentence) & set(dictionary['SUPERFLUOUS'].tolist())
    INTERESTING = set(sentence) & set(dictionary['INTERESTING'].tolist())


    return {'NEGATIVE':len(NEGATIVE),
    'POSITIVE':len(POSITIVE),
    'UNCERTAINTY':len(UNCERTAINTY),
    'LITIGIOUS':len(LITIGIOUS),
    'CONSTRAINING':len(CONSTRAINING),
    'SUPERFLUOUS':len(SUPERFLUOUS),
    'INTERESTING':len(INTERESTING)
    }





   
