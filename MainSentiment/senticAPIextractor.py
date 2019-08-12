
import json
import requests




URL = 'http://sentic.net/api/'
POLARITY_KEY = '868GAO/'
ASPECTS_KEY = '7553HJ/'



def extractPolarity(sentence):

	#print(URL+POLARITY_KEY+sentence)#contents= requests.get(URL+POLARITY_KEY+sentence)
	contents = requests.get(URL+POLARITY_KEY+sentence).json()
	return contents



def extractAspects(sentence):
	contents = requests.get(URL+ASPECTS_KEY+sentence).json()
	return contents


	


