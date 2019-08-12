from senticnet.senticnet import Senticnet
import pandas as pd
#import re

data=pd.read_csv('/home/andrea/Desktop/NLFF/sentiment/SenticTools/concept-parser/stanford-corenlp-python-final/AAPL.csv')
concepts=list()

#concepts=data['concepts'][0][0]
lista=list()
concepts=data['concepts'].head()
print(concepts)
'''
print(lista[0])
print('dopo')
lista[0]=lista[0].replace('[','')
lista[0]=lista[0].replace(']','')
lista[0]=lista[0].replace('\'','')
#lista[0]=lista[0].replace(',','')



lista[0]=lista[0].split(',')
#re.sub('[', '', data)
print(lista[0][0])

'''
polarity_value_result=list()
polarity_intense_result=list()
concept_info_result=list()

sn = Senticnet()
for concept in concepts:

	concept=concept.replace('[','')
	concept=concept.replace('u\'','')
	concept=concept.replace('_',' ')
	concept=concept.replace('...','')

	concept=concept.replace(']','')
	concept=concept.replace('\'','')
	concept=concept.split(',')
	print('internal loop')
	for item in concept:
		print('item:')
		print(item)
		item=str(item)
		#item='love'
		try:
			concept_info = sn.concept(item)
			polarity_value = sn.polarity_value(item)
			polarity_intense = sn.polarity_intense(item)
			print(polarity_value)
			moodtags = sn.moodtags(item)
			semantics = sn.semantics(item)
			sentics = sn.sentics(item)
			polarity_intense_result.append(polarity_intense)
			polarity_value_result.append(polarity_value)
			concept_info_result.append(concept_info)

		except:
			print('Concept not in senticnet')
print(polarity_value_result)