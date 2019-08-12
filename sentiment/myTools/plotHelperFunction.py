
##
## 	from plotHelperFunction import comparisonInfo
## 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime, timedelta


def comparisonInfo(indexFileCSV,  sentimentFileCSV, sentimentColumnName, timeColumnName, windowAverageMeanPrice, windowAverageMeanSentiment ):
	##------------------ INDEX ------------
	indexes = pd.read_csv(indexFileCSV)
	indexes.rename(columns={'Unnamed: 0':'date'}, inplace= True)
	print("Reading index file "+ indexFileCSV)
	for i in range(len(indexes)):
		indexes['date'].at[i]= datetime.strptime(indexes['date'].at[i], '%Y-%m-%d %H:%M:%S')

	#Normalizing indexes
	values = {'boll_ub': indexes["boll_ub"].mean(), 'boll_lb': indexes["boll_lb"].mean()}
	indexes = indexes.fillna(value=values)
	indexes = indexes.fillna(0)

	minSen = min(indexes['high'])
	maxSen = max(indexes['high'])
	for i in range(len(indexes)):
		indexes['high'].at[i] = (indexes['high'].at[i] - minSen)/(maxSen - minSen)
	
	#Media mobile anche sugli indici
	indexes['high'] = indexes['high'].rolling(window=windowAverageMeanPrice).mean()

	#print(indexes.head())


	## --------------- SENTIMENT --------------
	sentimentCSV = pd.read_csv(sentimentFileCSV)

	minSen = min(sentimentCSV[sentimentColumnName])
	maxSen = max(sentimentCSV[sentimentColumnName])
	for i in range(len(sentimentCSV)):
		sentimentCSV[sentimentColumnName].at[i] = 2*(sentimentCSV[sentimentColumnName].at[i] - minSen)/(maxSen - minSen) -1




	for i in range(len(sentimentCSV)):
		sentimentCSV[timeColumnName].at[i]= datetime.strptime(sentimentCSV[timeColumnName].at[i], '%Y-%m-%d %H:%M:%S')

	

	meanSentiment = sentimentCSV[sentimentColumnName].mean()
	sentimentCSV[sentimentColumnName] -= meanSentiment

	sentimentCSV[sentimentColumnName] = sentimentCSV[sentimentColumnName].rolling(window= windowAverageMeanSentiment).mean()

	## -------------- Alignment of datasets ------------

	initDate = max(indexes['date'][0], sentimentCSV[timeColumnName][0])
	finalDate = min(indexes['date'][len(indexes)-1], sentimentCSV[timeColumnName][len(sentimentCSV)-1])


	indexes = indexes[indexes['date'] >= initDate]
	indexes = indexes[indexes['date'] <= finalDate]
	sentimentCSV = sentimentCSV[sentimentCSV[timeColumnName] >= initDate]
	sentimentCSV = sentimentCSV[sentimentCSV[timeColumnName] <= finalDate]
	
	assert len(indexes['date'].tolist()) == len(sentimentCSV[timeColumnName].tolist())
	sentimentDateList = sentimentCSV[timeColumnName].tolist()
	indexesDateList = indexes['date'].tolist()

	for i in range(len(indexes)):
		assert sentimentDateList[i] == indexesDateList[i]

        
	print("Start: "+str(initDate))
	print("End: "+str(finalDate))
        


	#initTime = sentimentCSV[timeColumnName].at[0]
	#indexes = indexes[indexes['date'] >= initTime]
	#argmax = indexes['date'].idxmax()
	#if(indexes['date'][argmax] > sentimentCSV[timeColumnName][len(sentimentCSV)-1]):
	#	indexes = indexes[indexes['date'] <= sentimentCSV[timeColumnName][len(sentimentCSV)-1]]
	#else:
	#	sentimentCSV = sentimentCSV[sentimentCSV[timeColumnName] <= indexes['date'][argmax]]


	## ---------- Compute slope
	slopeStock = pd.Series(np.gradient(indexes['high']))
	accuracy = [a*b*1000 for a,b in zip(slopeStock,sentimentCSV[sentimentColumnName])]
	minSen = np.nanmin(accuracy)
	maxSen = np.nanmax(accuracy)

	for i in range(len(accuracy)):
		accuracy[i] = 2*(accuracy[i] - minSen)/(maxSen - minSen) -1


	return {'accuracy':accuracy,
			'slope':slopeStock.tolist(),
			'priceAverageMean':indexes['high'].tolist(),
			'sentimentAverageMean':sentimentCSV[sentimentColumnName].tolist(),
			'datesXaxis':sentimentDateList,
			'zeros':np.zeros(len(sentimentCSV))
			}









	