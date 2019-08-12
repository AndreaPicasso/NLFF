#
# FIRST: add google cloud as command


# export GOOGLE_APPLICATION_CREDENTIALS="/home/simone/Desktop/NLFF/GoogleNL/NLFF-3f52da65e963.json"

import argparse
import sys
import pandas as pd
import numpy as np
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import six
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import math
import json


# Instantiates a client

# # The text to analyze
# text = u'Apple\'s iPhone X has problems and it has significant consequences for the upcoming new models...'
# document = types.Document(
#     content=text,
#     type=enums.Document.Type.PLAIN_TEXT)

# # Detects the sentiment of the text
# response = client.analyze_sentiment(document=document)


# print('Text: {}'.format(text))
# print(response)
# sentiment = response.document_sentiment
# print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))





def entity_sentiment_text(text, client):
    """Detects entity sentiment in the provided text."""
    
    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(
        content=text.encode('utf-8'),
        type=enums.Document.Type.PLAIN_TEXT)

    # Detect and send native Python encoding to receive correct word offsets.
    encoding = enums.EncodingType.UTF32
    if sys.maxunicode == 65535:
        encoding = enums.EncodingType.UTF16

    result = client.analyze_entity_sentiment(document, encoding)

    return result






path = '/home/simone/Desktop/NLFF/intrinioDatasetUpdated/preprocessing/preprocessed/'
ticker = 'CSCO'
tickFiles = [f for f in listdir(path) if isfile(join(path, f))]
path +=ticker

news = pd.read_csv(path+'.csv')
count =1
while ticker+str(count)+'.csv' in tickFiles:
    print(path+str(count))
    newsTemp = pd.read_csv(path +str(count)+'.csv')
    news = pd.concat([news, newsTemp])
    count+=1
news.drop_duplicates(subset=['PUBLICATION_DATE'], inplace=True)
news = news.sort_values(by=['PUBLICATION_DATE'])
news = news.reset_index(drop=True)
    
aapl = news
for i in range(len(aapl)):
    aapl.at[i,'PUBLICATION_DATE'] =datetime.strptime(aapl['PUBLICATION_DATE'][i], '%Y-%m-%d %H:%M:%S +%f')

##
# Selected intervall for trial: from 1/1/2018 to 1/4/2018
##2017-07-02 23:00:00 

initDate = datetime(2017, 5, 22, 0, 0, 0)
endDate = datetime(2018, 6, 21, 0, 0, 0)

aapl.drop(aapl[aapl.PUBLICATION_DATE <= initDate ].index, inplace=True)
aapl.drop(aapl[aapl.PUBLICATION_DATE > endDate ].index, inplace=True)
#aapl.drop(aapl[aapl.PUBLICATION_DATE > datetime(2018, 1, 1, 0, 0, 0)].index, inplace=True)

count = 0
for index, row in aapl.iterrows():
	text = str(row['TITLE'])+'.\n'+str(row['SUMMARY'])
	count += math.ceil(len(str(text))/1000)

print('Interval: ['+str(initDate)+','+str(endDate)+']')
x = input('num news: '+str(count)+' continue?(Y/n): ')
assert x == 'Y'



aapl['gReply'] = ''
#f = open('googleSentiment.csv', 'a')
fJSON = open(str(ticker)+'_googleSentimentJSON.csv', 'a')


client = language.LanguageServiceClient()
count = 0
for i, row in aapl.iterrows():
	text = str(row['TITLE'])+'.\n'+str(row['SUMMARY'])
	count += math.ceil(len(str(text))/1000)
	print(count)
	# if count > 4990:
	# 	print('\n\n\n ~~~~~~~~~~~~~ LIMIT REACHED ~~~~~~~~~~~~~~~~')
	# 	break
	text = str(row['TITLE'])+'.\n'+str(row['SUMMARY'])
	gResult = entity_sentiment_text(text, client)

	data = {}
	entities = list()
	data['entities'] = entities
	for entity in gResult.entities:
	    entityJ = {}
	    entities.append(entityJ)
	    mentionsJ = list()

	    entityJ['name'] = entity.name if entity.name is not None else ''
	    entityJ['salience'] = entity.salience if entity.salience is not None else ''
	    entityJ['sentiment'] = {'magnitude': entity.sentiment.magnitude, 'score':entity.sentiment.score}
	    entityJ['type'] = entity.type if entity.type is not None else ''
	    entityJ['mentions'] = mentionsJ    
	    

	    for mention in entity.mentions:
	        sentiment = {}
	        text = {}
	        text['content'] = mention.text.content
	        text['begin_offset'] = mention.text.begin_offset
	        sentiment['magnitude'] = mention.sentiment.magnitude
	        sentiment['score'] = mention.sentiment.score
	        mentionsJ.append({'text':text, 'type':mention.type, 'sentiment':sentiment})

	data['entities'] = entities
	data['language'] = gResult.language
	json_data = json.dumps(data)

	gResult = str(gResult)    
	gResult = gResult.replace('\n','')
	gResult = gResult.replace('\t','')


	#aapl.at[i,'gReply'] = gResult
	#f.write('"'+str(row['PUBLICATION_DATE'])+'" ; "'+gResult+'"\n\n')
	fJSON.write('"'+str(row['PUBLICATION_DATE'])+'" ; "'+json_data+'"\n')


    
#f.close()
fJSON.close()
print('Request done: '+str(count))