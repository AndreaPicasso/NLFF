
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from math import sqrt
import matplotlib.pyplot as plt

#tf.logging.set_verbosity(tf.logging.INFO)



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


    def get_train_test_set(test_percentage=0.3):
        idx_split = math.floor(len(Data.pos)*(1-test_percentage))

        train_pos = Data.pos[:idx_split]
        train_neg = Data.neg[:idx_split]
        train_y = Data.Y[:idx_split]
        test_pos = Data.pos[idx_split:]
        test_neg = Data.neg[idx_split:]
        test_y = Data.Y[idx_split:]

        return (train_pos, train_neg, train_y), (test_pos, test_neg, test_y)



    def load_data(ticker='AAPL', momentum_window=30, newsTimeToMarket =0, X_window_average=40, set_verbosity=True):
        X_path = '../tensorflow_model/for_server/SentimentSingleNewsFullNoNorm/'+str(ticker)+'.csv'
        Y_path = '../tensorflow_model/for_server/DataSetIndexes/indexes'+str(ticker)+'.csv'
        
        x = pd.read_csv(X_path)
        x.drop('Unnamed: 0', axis=1, inplace=True)
        x = x.rename(index=str, columns={"initTime": "PUBLICATION_DATE"})
        x = x.sort_values(by=['PUBLICATION_DATE'])
        x = x.reset_index(drop=True)
        y =  pd.read_csv(Y_path)
        for i, row in x.iterrows():
            x.at[i,'PUBLICATION_DATE'] =datetime.strptime(x['PUBLICATION_DATE'][i], '%Y-%m-%d %H:%M:%S') + timedelta(hours=newsTimeToMarket)

        momentum_window = 30
        y = y.rename(index=str, columns={"Unnamed: 0": "DATE"})

        for i, row in y.iterrows():
            y['DATE'].at[i] = datetime.strptime(y['DATE'][i], '%Y-%m-%d %H:%M:%S') 
        z = list()
        for i in range(0,y.shape[0]-momentum_window):
            z.append((y['close'][i] - y['close'][i-momentum_window])/y['close'][i])

        y = y.reset_index(drop=True)
        y.drop(np.arange(y.shape[0]-momentum_window, y.shape[0]), inplace=True)
        y = y.reset_index(drop=True)
        y['labels'] = [sign(entry) for entry in z]
        min_max_scaler = preprocessing.MinMaxScaler()

        initDate = max(y['DATE'][0], x['PUBLICATION_DATE'][0])
        finalDate = min(y['DATE'][len(y)-1], x['PUBLICATION_DATE'][len(x)-1])
        i = 0
        j = 0

        close = []
        labels = []
        pos = []
        neg = []

        dates = []
        # ALLINEAMENTO INIZIO
        while(y['DATE'][j] < initDate):
            j+=1
        while(x['PUBLICATION_DATE'][i] < initDate):
            i+=1

        while(x['PUBLICATION_DATE'][i] < finalDate and y['DATE'][j] < finalDate ):
            timeSlotPos = list()
            timeSlotNeg = list()
            while(i<len(x)-1 and y['DATE'][j] > x['PUBLICATION_DATE'][i]):
                timeSlotPos.append(x['POSITIVE'][i])
                timeSlotNeg.append(x['NEGATIVE'][i])
                i+=1
            if(len(timeSlotPos)==0):
                timeSlotPos.append(0)
                timeSlotNeg.append(0)
            pos.append(np.mean(np.asarray(timeSlotPos), axis=0))  
            neg.append(np.mean(np.asarray(timeSlotNeg), axis=0))   
            
            close.append(y['close'][j])
            labels.append(y['labels'][j])
            dates.append(str(y['DATE'][j].year)+'/'+str(y['DATE'][j].month))
            
            j+=1

        pos = np.convolve(np.asarray(pos), np.repeat(1.0, X_window_average)/X_window_average, 'same')
        neg = np.convolve(np.asarray(neg), np.repeat(1.0, X_window_average)/X_window_average, 'same')


        Data.pos = pos
        Data.neg = neg
        Data.Y = labels






class ModelSelection():
        

    def modelSelectionFixedTTM(ticker='AAPL'):
        print('\n\n\n==================== '+str(ticker)+' ==================== \n\n\n')

        test_accs = []
        MCCs = []
        MCCsReal = []
        TP = []
        TN = []
        FP = []
        FN = []
        Ttm_range = [0, 7, 14, 21,28, 35, 70, 105, 210]
        for ttm in Ttm_range:            
            Data.load_data(ticker=ticker, momentum_window=30, newsTimeToMarket =ttm, X_window_average=30, set_verbosity=False)
            (train_pos, train_neg, train_y), (test_pos, test_neg, test_y) = Data.get_train_test_set()


            # best_MCC = 0
            # best_b = 0
            # for bias in np.linspace(-1,1,20):
            #     yhat = list()
            #     for i in range(len(train_y)):
            #         yhat.append(1 if train_pos[i]+bias >= train_neg[i] else 0)

            #     cm = confusion_matrix(train_y, yhat)
            #     tn, fp, fn, tp = cm.ravel()

            #     denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
            #     curr_MCC = 0 if denom== 0 else (tp*tn -fp*fn)/sqrt(denom)
            #     if(curr_MCC > best_MCC):
            #         best_MCC = curr_MCC
            #         best_b = bias

            # bias = best_b
            bias = np.mean(train_neg) - np.mean(train_pos)
            yhat = list()
            for i in range(len(test_y)):
                yhat.append(1 if test_pos[i]+bias >= test_neg[i] else 0)


            
            cm = confusion_matrix(test_y, yhat)
            tn, fp, fn, tp = cm.ravel()
            denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
            MCCsReal.append(0 if denom== 0 else (tp*tn -fp*fn)/sqrt(denom) )
            TP.append(tp)
            TN.append(tn)
            FN.append(fn)
            FP.append(fp)
            test_accs.append((tp+tn)/(tp+tn+fp+fn))


            if(tp + fp == 0):
                tp = 1
            if(tp + fn == 0):
                tp = 1
            if(tn + fp == 0):
                tn = 1
            if(tn + fn == 0):
                tn = 1

            MCCs.append((tp*tn -fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))



        #print(ticker)
        #print('best b: '+str(bias))

        #print('Ttm_range, '+str(Ttm_range))
        print('test acc,'+str(ticker)+', '+str(test_accs))
        #print('MCC,'+str(ticker)+', '+str(MCCs))
        print('MCC_R,'+str(ticker)+', '+str(MCCsReal))
        print('TN,'+str(ticker)+', '+str(TN))
        print('FP,'+str(ticker)+', '+str(FP))
        print('FN,'+str(ticker)+', '+str(FN))
        print('TP,'+str(ticker)+', '+str(TP))

        



tickers = ['AAPL','AMZN','GOOGL','MSFT','FB','INTC','CSCO','CMCSA','NVDA','NFLX']       
for tic in tickers:
    ModelSelection.modelSelectionFixedTTM(ticker=tic)
