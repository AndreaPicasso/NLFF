import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt




# . LOAD DATA DEGLI SKIP THROUGH VECTOR E DEL MERCATO
# .
# .
# .



X_COLUMN_NAMES = ['PUBLICATION_DATE', 'EMBEDDING']
#Y_COLUMN_NAMES = ['DATE', 'close', 'high','low', 'close','volume','close_12_ema','close_26_ema','macd','macds','macdh','macd','macds', 'boll_ub', 'boll_lb', 'rsi_6','rsi_12','vr_6_sma','wr_10','wr_6']


## QUESTO DATASET HA GIA AVG DEL SENTIMENT OGNI ORA
X_path = '/home/simone/Desktop/NLFF/sentiment/myTools/wordCount+KSVM/SentimentNews/AAPL_tot.csv'
Y_path = '../../../DataSetIndexes/indexesAAPL.csv'
def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        #return -1
        return 0
    
class Data():
    X = []
    Y = []

    def get_train_test_set(test_percentage=0.3):
        idx_split = math.floor(len(Data.X)*(1-test_percentage))

        train_x = Data.X[:idx_split]
        train_y = Data.Y[:idx_split]
        test_x = Data.X[idx_split:]
        test_y = Data.Y[idx_split:]

        return (train_x, train_y), (test_x, test_y)



    def get_cross_validation_train_dev_set(test_percentage=0.3, k_fold = 10,  dev_num_points=100):

        # https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection/14109#14109


        # fold = list(train_i , dev_i)
        # train_i = (train_x, train_y) 
        (train_x, train_y), _  = Data.get_train_test_set(test_percentage=test_percentage)
        m = int(len(train_x))
        samples_per_fold = int( (m - dev_num_points) / k_fold)
        print('Fold lenght: '+str(samples_per_fold))
        fold = list()
        index = 0
        while(len(fold) < k_fold):
            fold.append(((train_x[index:index+samples_per_fold], train_y[index:index+samples_per_fold]),
                    (train_x[index+samples_per_fold:index+samples_per_fold+dev_num_points], train_y[index+samples_per_fold:index+samples_per_fold+dev_num_points])))
            index += samples_per_fold

        return fold





    def load_data( momentum_window=30, X_window_average=None, newsTimeToMarket = 0):

        print('Reading dataset...')
        x = pd.read_csv(X_path)
        x = x.rename(index=str, columns={"initTime": "PUBLICATION_DATE"})
        #cambio l'ordine dalla piu vecchia alla piu recente
        print('Ordering dataset...')
        x = x.sort_values(by=['PUBLICATION_DATE'])
        x = x.reset_index(drop=True)
        
        if(X_window_average != None):
            print('Moving average..')
            x['CONSTRAINING'] = x['CONSTRAINING'].rolling(window=X_window_average,center=False).mean()
            x['LITIGIOUS'] = x['LITIGIOUS'].rolling(window=X_window_average,center=False).mean()
            x['NEGATIVE'] = x['NEGATIVE'].rolling(window=X_window_average,center=False).mean()
            x['POSITIVE'] = x['POSITIVE'].rolling(window=X_window_average,center=False).mean()
            x['UNCERTAINTY'] = x['UNCERTAINTY'].rolling(window=X_window_average,center=False).mean()
            x.drop(np.arange(X_window_average-1), inplace=True)
            x = x.reset_index(drop=True)
        
        #Normalizzo
        min_max_scaler = preprocessing.MinMaxScaler()
        x[['CONSTRAINING', 'LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY']] = min_max_scaler.fit_transform(x[['CONSTRAINING', 'LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY']].values)
            

        for i, row in x.iterrows():
            x.at[i,'PUBLICATION_DATE'] =datetime.strptime(x['PUBLICATION_DATE'][i], '%Y-%m-%d %H:%M:%S') + timedelta(minutes=newsTimeToMarket)

            
            
            
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

        
        # plt.figure(figsize=(20,10))
        # plt.plot(np.arange(0, len(x)), x['CONSTRAINING'],'b')
        # plt.plot(np.arange(0, len(x)), x['LITIGIOUS'],'g')
        # plt.plot(np.arange(0, len(x)), x['NEGATIVE'],'m')
        # plt.plot(np.arange(0, len(x)), x['POSITIVE'],'c')
        # plt.plot(np.arange(0, len(x)), x['UNCERTAINTY'],'k')
        # plt.ylabel('INPUT')
        # plt.xlabel('time')
        # plt.plot()

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
            if not (y['DATE'][j] >= x['PUBLICATION_DATE'][i] and y['DATE'][j-1] < x['PUBLICATION_DATE'][i]):
            	print("y['DATE']["+str(j)+"]: "+str(y['DATE'][j])+" x["+str(i)+"] : " +str(x['PUBLICATION_DATE'][i]) + " y['DATE']["+str(j-1)+"]: "+str(y['DATE'][j-1]))
            	assert False
            X.append([x['CONSTRAINING'][i],x['LITIGIOUS'][i],x['NEGATIVE'][i],x['POSITIVE'][i],x['UNCERTAINTY'][i]]) 
            i+=1
            Y.append(y['close'][j])
            j+=1


        Xsequence = X
        # Xsequence = list()
        # Xsequence.insert(0, X)
        # n = len(X)
        # for i in range(1,n):
        # 	X = X[:-1]
        # 	X.insert(0, [0]*5)
        # 	Xsequence.insert(0, X)
        	
        	# Xsequence.append(np.asarray(X[:i+1]))

        Data.X = np.nan_to_num(np.asarray(Xsequence))
        Data.Y = np.nan_to_num(np.asarray(Y))