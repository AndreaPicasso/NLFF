import pandas as pd
import numpy as np
import tensorflow as tf
import math
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#import matplotlib.pyplot as plt


skip_vector_dim = 7
n_y = 1 #Numero di output, Per ora sali / scendi poi metteremo neutrale


def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        #return -1
        return 0
    
class Data():
    X = []
    Y = []
    dates = []

    def get_train_test_set(test_percentage=0.3):
        idx_split = math.floor(len(Data.X)*(1-test_percentage))
        print('Train\t'+str(Data.dates[0])+'\t'+str(Data.dates[idx_split-1])+'\t'+str(idx_split))
        print('Test\t'+str(Data.dates[idx_split-1])+'\t'+str(Data.dates[-1])+'\t'+str(len(Data.X) - idx_split))
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
        # print('Fold lenght: '+str(samples_per_fold))
        fold = list()
        index = 0
        while(len(fold) < k_fold):
            # Training folds of equal length:
#           fold.append(((train_x[index:index+samples_per_fold], train_y[index:index+samples_per_fold]),
#                 (train_x[index+samples_per_fold:index+samples_per_fold+dev_num_points], train_y[index+samples_per_fold:index+samples_per_fold+dev_num_points])))
            fold.append(((train_x[0:index+samples_per_fold], train_y[0:index+samples_per_fold]),
                (train_x[index+samples_per_fold:index+samples_per_fold+dev_num_points], train_y[index+samples_per_fold:index+samples_per_fold+dev_num_points])))
            index += samples_per_fold

        return fold





    def load_data(ticker, momentum_window=30, X_window_average=None, news_per_hour = 10, newsTimeToMarket = 0, set_verbosity=True):
        X_path = 'SentimentSingleNewsFullNoNorm/'+str(ticker)+'.csv'
        Y_path = 'DataSetIndexes/indexes'+str(ticker)+'.csv'
        
        
        if(set_verbosity):
            print('Reading dataset...')
            
        x = pd.read_csv(X_path)
        x.drop('Unnamed: 0', axis=1, inplace=True)

        x = x.rename(index=str, columns={"initTime": "PUBLICATION_DATE"})
    
        ##cambio l'ordine dalla piu vecchia alla piu recente
        if(set_verbosity):
            print('Ordering dataset...')
        x = x.sort_values(by=['PUBLICATION_DATE'])
        x = x.reset_index(drop=True)
        
        if(X_window_average != None):
            if(set_verbosity):
                print('Moving average..')
            x['CONSTRAINING'] = x['CONSTRAINING'].rolling(window=X_window_average,center=False).mean()
            x['LITIGIOUS'] = x['LITIGIOUS'].rolling(window=X_window_average,center=False).mean()
            x['NEGATIVE'] = x['NEGATIVE'].rolling(window=X_window_average,center=False).mean()
            x['POSITIVE'] = x['POSITIVE'].rolling(window=X_window_average,center=False).mean()
            x['UNCERTAINTY'] = x['UNCERTAINTY'].rolling(window=X_window_average,center=False).mean()
            x['SUPERFLUOUS'] = x['SUPERFLUOUS'].rolling(window=X_window_average,center=False).mean()
            x['INTERESTING'] = x['INTERESTING'].rolling(window=X_window_average,center=False).mean()
            
            x.drop(np.arange(X_window_average-1), inplace=True)
            x = x.reset_index(drop=True)
        
        #Normalizzo
        min_max_scaler = preprocessing.MinMaxScaler()
        x[['CONSTRAINING', 'LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'
           ,'SUPERFLUOUS','INTERESTING']] = min_max_scaler.fit_transform(x[['CONSTRAINING', 'LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY','SUPERFLUOUS','INTERESTING']].values)

            

        for i, row in x.iterrows():
            x.at[i,'PUBLICATION_DATE'] =datetime.strptime(x['PUBLICATION_DATE'][i], '%Y-%m-%d %H:%M:%S') + timedelta(hours=newsTimeToMarket)

                        
        y = pd.read_csv(Y_path)
        y = y.rename(index=str, columns={"Unnamed: 0": "DATE"})

        #PER ORA SCARTO GLI INDICI, POI SARA' DA METTERLI DENTRO X
        #y = y['DATE', 'close']
        for i, row in y.iterrows():
            y['DATE'].at[i] = datetime.strptime(y['DATE'][i], '%Y-%m-%d %H:%M:%S') 

        z = list()
        if(set_verbosity):
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
        dates = list()
        if(set_verbosity):
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
               timeSlotX.append([x['CONSTRAINING'][i],x['LITIGIOUS'][i],x['NEGATIVE'][i],x['POSITIVE'][i],x['UNCERTAINTY'][i],x['SUPERFLUOUS'][i],x['INTERESTING'][i]])

               i+=1


            # Da len(timeslot) dobbiamo ricondurci ad avere news_per_hour numero di news
            # Random sampling se sono troppe:
            if(len(timeSlotX) > news_per_hour):
                #timeSlotX = np.random.choice(timeSlotX, news_per_hour, replace=False)
                selectedIndexes = np.random.choice(range(0,len(timeSlotX)-1), news_per_hour, replace=False).tolist()
                timeSlotX =  [timeSlotX[index] for index in selectedIndexes]
                
            # Replicazione news se sono troppo poche
            else:
                if(len(timeSlotX) < news_per_hour):
                    index = 0
                    #Se non e presente manco una news riempi di zeri
                    if(len(timeSlotX) == 0):
                        timeSlotX.append([0] * skip_vector_dim)
                        
                    numNews = len(timeSlotX)

                    while(len(timeSlotX) < news_per_hour):
                        timeSlotX.append(timeSlotX[index%numNews])
    
            X.append(timeSlotX)   
            Y.append(y['close'][j])
            dates.append(y['DATE'][j])
            j+=1
        
        assert len(X) == len(Y)
        Data.X = X
        Data.Y = Y
        Data.dates = dates


tickers = ['AAPL','AMZN','GOOGL','MSFT','FB','INTC','CSCO','CMCSA','NVDA','NFLX'] 
for ticker in tickers:
    print(ticker)
    Data.load_data(ticker= ticker, X_window_average=30, momentum_window=30, newsTimeToMarket = 0, set_verbosity=False)
    Data.get_train_test_set()