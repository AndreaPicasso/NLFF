import urllib.request
import json


URL = 'http://sentic.net/api/'
POLARITY_KEY = '868GAO/'
ASPECTS_KEY = '7553HJ/'



def extractPolarity(sentence):

	contents = json.loads(urllib.request.urlopen(URL+POLARITY_KEY+sentence).read())
	return contents



def extractAspects(sentence):
	contents = json.loads(urllib.request.urlopen(URL+ASPECTS_KEY+sentence).read())
	return contents


	


