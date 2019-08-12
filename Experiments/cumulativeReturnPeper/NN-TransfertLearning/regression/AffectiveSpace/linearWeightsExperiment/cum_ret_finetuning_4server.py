import sys
sys.path.append('../../../../../../Experiments/cumulativeReturnPeper')
import os.path
import pickle


import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from math import sqrt
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,LeakyReLU
from keras.constraints import max_norm
from keras import optimizers,regularizers
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, mean_squared_error

from technicalSignals import Indicators


tickers=['AAPL','AMZN','GOOGL','MSFT','FB','INTC','CSCO','CMCSA','NVDA','NFLX']
TREND_WINDOWs = [(1,50)]
kind_of_dataset = 'AffectiveSpace'
NN_INPUT_DIM = 717

class DatasetManager:
    def __init__(self):
        X_raw = None
        Y_raw = None
        Y = None
        X = None
    
    def load_dataset(self, ticker, kind, technicalFeatures=False):
        types = {'Summary': '../../../../../../intrinioDatasetUpdated/SentimentFullAggregatedHourly/',
            'AffectiveSpace': '../../../../../../AffectiveSpace/Aggregated_AffectSummary_dataset/',
            }
        news =  pd.read_csv(types[kind]+ticker+'.csv')
        price = pd.read_csv('../../../../../../indexes/indexes'+ticker+'.csv')
        price = price.rename(index=str, columns={"date": "DATE"})
        news = news.rename(index=str, columns={"initTime": "DATE"})
        news = news.drop(['Unnamed: 0'], axis=1)
        news['DATE'] = [datetime.strptime(row, '%Y-%m-%d %H:%M:%S') for row in news['DATE']]
        # This datased is already GMT+0
        price['DATE'] = [datetime.strptime(row, '%Y-%m-%d %H:%M:%S') for row in price['DATE']]
        if(technicalFeatures):
            price['mom_30'] = Indicators.momentum(price, 30)
            price['mom_50'] = Indicators.momentum(price, 50)
            price['mom_100'] = Indicators.momentum(price, 100)
            price['mom_150'] = Indicators.momentum(price, 150)
            price['SMA_30'] = Indicators.SMA(price, 30)
            price['SMA_50'] = Indicators.SMA(price, 50)
            price['SMA_100'] = Indicators.SMA(price, 100)
            price['SMA_150'] = Indicators.SMA(price, 150)
            price['in_BBands'] = Indicators.inBBands(price)
            price['eccessVolumes'] = Indicators.eccessOfVolumes(price)


        #ALLIGNMENT
        initDate = max(news['DATE'][0], datetime(2017, 5, 22, 0, 0, 0))
        finalDate = min(news['DATE'][len(news)-1],datetime(2018, 6, 20, 0, 0, 0))
        news.drop(news[news.DATE > finalDate].index, inplace=True)
        news.drop(news[news.DATE < initDate].index, inplace=True)
        news = news.reset_index(drop=True)
        price.drop(price[price.DATE > finalDate].index, inplace=True)
        price.drop(price[price.DATE < initDate].index, inplace=True)
        price = price.reset_index(drop=True)
        assert len(price) == len(news)
        # FEATURES
        sentiment = news.drop(['DATE'], axis=1)
        X = sentiment
        for window in [5,10,15,20,30,50]:
            temp = sentiment.rolling(window).mean()
            temp.columns = temp.columns +'_'+str(window)
            X = pd.concat([X, temp],axis=1)
        if(technicalFeatures):   
            technical_features = ['mom_30','mom_50','mom_100','mom_150',
                                  'SMA_30','SMA_50','SMA_100','SMA_150','in_BBands','eccessVolumes']
            X = pd.concat([X, price[technical_features]],axis=1)

            
        #NORMALIZATION:
        with open('min_max_scaler_trained_allTickers_VolumeFeature.pickle', 'rb') as handle:
            min_max_scaler = pickle.load(handle)
        X = np.nan_to_num(np.asarray(X, dtype=float))
        X = np.asarray(min_max_scaler.transform(X))
        self.X_raw = X
        self.Y_raw = price

    def get_dataset_for_trend(self, init, finish, perc_train = 0.7):
        y = list()
        x = list()
        dates = list()
        price = self.Y_raw
        for i in range(abs(init),len(price)-finish):
            cumulative_return =  (price.iloc[i+finish]['open']-price.iloc[i+init]['open'])/price.iloc[i+init]['open']
            y.append(cumulative_return)
            dates.append(price.iloc[i]['DATE'])
            x.append(self.X_raw[i])
        
        y = np.array(y)
        x = np.array(x)
        self.X = x
        self.Y = y
        nt=math.ceil(len(x)*perc_train)
        x_tv = x[:nt]
        y_tv = y[:nt]
        x_test = x[nt:]
        y_test = y[nt:]
        dates_test = dates[nt:]
        return (x_tv,y_tv),(x_test,y_test),dates_test
    
    
    def get_dataset_for_trend_all_tickers(self, init, finish,kind, perc_train = 0.7, technicalFeatures=False):
        x_tv_all = []
        y_tv_all = []
        x_test_all = []
        y_test_all = []
        dates_test_prev = None
        for ticker in tickers:
            self.load_dataset(ticker, kind, technicalFeatures)
            (x_tv,y_tv),(x_test,y_test),dates_test = ds.get_dataset_for_trend(init, finish, perc_train = 0.7)
            if(dates_test_prev):
                assert dates_test == dates_test_prev #I'm not secure about this constraint but otherwise which dates I will output?
            x_tv_all += x_tv.tolist()
            y_tv_all += y_tv.tolist()
            x_test_all += x_test.tolist()
            y_test_all += y_test.tolist()
        x_tv_all = np.asarray(x_tv_all)
        y_tv_all = np.asarray(y_tv_all)
        x_test_all = np.asarray(x_test_all)
        y_test_all = np.asarray(y_test_all)
        return (x_tv_all,y_tv_all),(x_test_all,y_test_all), dates_test



def weighted_MSE(y_true, y_pred):
    weights = K.pow(y_true,2)
    m = K.min(weights)
    M = K.max(weights)
    weights = (weights-m)/(M-m)
    return  K.mean(K.pow(y_true - y_pred, 2)*weights)

def get_pretrained_model(init, finish):
    json_file = open('pretraining_weights/nn_model_'+str(init)+'_'+str(finish)+'_vol.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    nn_model = model_from_json(loaded_model_json)
    nn_model.load_weights('pretraining_weights/nn_model_pretrained_weights_all_tickers_REGR_'+str(init)+'_'+str(finish)+'_vol.h5')
    #All layers not trainable except last one
    i=0
    while nn_model.get_layer(index=i) != nn_model.get_layer(index=-1):
        nn_model.get_layer(index=i).trainable = False
        i +=1
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    nn_model.compile(optimizer=opt,loss='mse', metrics=[weighted_MSE])
    return nn_model


def reset_weights(model):
    session=K.get_session()
    for layer in model.layers:
        if(hasattr(layer,'kernel_initializer')):
            layer.kernel.initializer.run(session=session)
            
# def plot_hystory(history,y_pred,y_test,l2,drop,n_units):
#     f, axarr = plt.subplots(2)
#     f.set_figheight(7)
#     f.set_figwidth(10)
#     axarr[0].plot(y_test,'g', label='y_test')
#     axarr[0].plot(y_pred,'r', label='y_pred')
#     if history:
#         axarr[1].semilogy(history.history['loss'],'g--',label='loss_train')
#         axarr[1].semilogy(history.history['val_loss'],'r--',label='loss_val')
#         axarr[1].semilogy(history.history['val_weighted_MSE'],'b:',label='loss_val_WITH_W')
#         axarr[1].semilogy(history.history['weighted_MSE'],'g:',label='loss_WITH_W')
#         print('Min val loss: ', min(history.history['val_loss']))
#         print('Val loss: ', history.history['val_loss'][-1])
#     axarr[1].legend()   
#     axarr[0].legend()   
#     axarr[0].set_title('l2: '+str(l2)+' drop: '+str(drop)+' n_units: '+str(n_units))   
#     plt.show()


            
            
def best_num_epoch(x_tv,y_tv):
    nn_model = get_pretrained_model(init, finish)
    trainpoint=math.floor(len(x_tv)*0.7)
    x_train=x_tv[0:trainpoint]
    y_train=y_tv[0:trainpoint]
    x_val=x_tv[trainpoint:]
    y_val=y_tv[trainpoint:]
    weights = np.power(y_train,2)
    m = min(weights)
    M = max(weights)
    weights = (weights-m)/(M-m)
    history = nn_model.fit(x_train, y_train, epochs = 200,batch_size =256, verbose=0, 
                           validation_data=(x_val, y_val),shuffle=True,
                           sample_weight = weights)
    best_epoch = np.argmin(np.convolve(history.history['val_weighted_MSE'], np.ones((4,))/4, mode='valid'))
                
    return best_epoch

losses = list()
# ALL TICKERS
for ticker in tickers:
    for (init, finish) in TREND_WINDOWs:
        print('\n\n\n==================== ',ticker,' trend: ',init,' ',finish, ' ==================== \n\n')
        ds = DatasetManager()
        ds.load_dataset(ticker = ticker, kind = kind_of_dataset, technicalFeatures=True)
        (x_tv,y_tv),(x_test,y_test),dates_test = ds.get_dataset_for_trend(init, finish, perc_train = 0.7)
        nn_model = get_pretrained_model(init, finish)
        best_epochs = best_num_epoch(x_tv,y_tv)
        print('Epochs: ',best_epochs)
        if best_epochs > 0:
            weights = np.power(y_tv,2)
            m = min(weights)
            M = max(weights)
            weights = (weights-m)/(M-m)
            best_num_epoch(x_tv,y_tv)
            history = nn_model.fit(x_tv, y_tv, epochs = best_epochs, batch_size =256, verbose=0,
                                   validation_data=(x_test, y_test),shuffle=True, sample_weight = weights)
        else:
            history = None
        y_pred = nn_model.predict(x_test, batch_size=256, verbose=0)
        nn_model.save_weights('fineTuning_weights/nn_model_finetuned_weights_'+ticker+'_REGR_'+str(init)+'_'+str(finish)+'.h5')   
        np.savetxt('test_predictions/'+ticker+'_'+str(init)+'_'+str(finish)+'.csv', y_pred, delimiter=",")

        print('==== Test ===')
        #plot_hystory(history,y_pred,y_test,'','','')  
        
        losses.append(history.history['loss'])
        losses.append(history.history['val_loss'])
losses = np.asarray(losses)
np.savetxt('losses.csv', losses, delimiter=",")
