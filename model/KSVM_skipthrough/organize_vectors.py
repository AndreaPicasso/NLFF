
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from datetime import datetime, timedelta

momentum_window=30
X_window_average=None
newsTimeToMarket = 20
X_COLUMN_NAMES = ['PUBLICATION_DATE', 'EMBEDDING']
X_path = '/home/simone/Desktop/AAPL_EMBEDDING_2.json'
Y_path = '../../DataSetIndexes/indexesAAPL.csv'

def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        #return -1
        return 0

X = []
Y = []	




print('Reading dataset...')
x = pd.read_json(X_path)
#cambio l'ordine dalla piu vecchia alla piu recente
print('Ordering dataset...')
x = x.sort_values(by=['PUBLICATION_DATE'])
x = x.reset_index(drop=True)
print(x.head())

if(X_window_average != None):
    x['EMBEDDING'] = x['EMBEDDING'].rolling(window=X_window_average,center=False).mean()
    x.drop(np.arange(X_window_average-1), inplace=True)
    x = x.reset_index(drop=True)




for i, row in x.iterrows():
    x.at[i,'PUBLICATION_DATE'] =datetime.strptime(x['PUBLICATION_DATE'][i], '%Y-%m-%d %H:%M:%S +%f') + timedelta(minutes=newsTimeToMarket)
    x.at[i, 'EMBEDDING']= x['EMBEDDING'][i][0]

y = pd.read_csv(Y_path)
y = y.rename(index=str, columns={"Unnamed: 0": "DATE"})

#PER ORA SCARTO GLI INDICI, POI SARA' DA METTERLI DENTRO X
#y = y['DATE', 'close']
for i, row in y.iterrows():
    y['DATE'].at[i] = datetime.strptime(y['DATE'][i], '%Y-%m-%d %H:%M:%S') 

z = list()
print('y(t) - y(t-1) ...')

#calcolo differenza price(t) - price(t-window)
for i in range(0,momentum_window):
    z.append(523) #Valore impossibile per fare drop successivamente  
for i in range(momentum_window,y.shape[0]):
    z.append(sign(y['close'][i] - y['close'][i-momentum_window]))
y['close'] = z

y = y[y['close'] != 523] #Ellimino primi valori per momentum window



X = list()
Y = list()

print('Alligning dataset and constructing cube..')

initDate = max(y['DATE'][0], x['PUBLICATION_DATE'][0])
finalDate = min(y['DATE'][len(y)-1], x['PUBLICATION_DATE'][len(x)-1])
i = 0
j = 0

# ALLINEAMENTO INIZIO
while(y['DATE'][j] < initDate):
    j+=1
while(x['PUBLICATION_DATE'][i] < initDate):
    i+=1

while(x['PUBLICATION_DATE'][i] < finalDate and y['DATE'][j] < finalDate ):
    timeSlotX = list()
    while(i<len(x)-1 and y['DATE'][j] > x['PUBLICATION_DATE'][i]):
        timeSlotX.append(x['EMBEDDING'][i]) 
        i+=1
        if(i%1000 == 0):
            print(str(i)+ '/' + str(x.shape[0]))


    if(len(timeSlotX) == 0):
    	timeSlotX.append([0] * 2400)
    timeSlotX = np.mean(np.asarray(timeSlotX), axis=0)
    X.append(timeSlotX)   
    Y.append(y['close'][j])
    j+=1



X = np.asarray(X)
Xdf = pd.DataFrame()
for i in range(0, 2400):
	Xdf[str(i)] = X[:,i]
Xdf.to_csv('X.csv')


Y = np.asarray(Y)
Y = pd.DataFrame(data = Y)
Y.to_csv('Y.csv')

print('Done')



