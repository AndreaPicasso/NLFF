from __future__ import division
from imblearn.metrics import geometric_mean_score
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
np.random.seed(13)
import time
from datetime import datetime, timedelta
from sklearn import svm
import pandas as pd

import matplotlib.pyplot as plt
import math

from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from stockstats import StockDataFrame 
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data

files=['AAPL','AMZN','PEP','GOOGL','MSFT','FB','INTC','CSCO','CMCSA','NVDA','NFLX','BKNG','ADBE','AMGN','TXN','AVGO','PYPL','GILD','COST','QCOM']       
def smote(x,y):
    X_resampled, y_resampled = SMOTE().fit_sample(x, y)
    #print('check',sum(y_resampled)/len(y_resampled))
    return X_resampled,y_resampled
def balance(x,y):
    posindex=np.where( y == 1 )
    negindex=np.where( y == 0 )
    xt=[]
    yt=[]
    yindex=[]
    nindex=min(len(posindex[0]),len(negindex[0]))

    #for i in range(1,nindex):
    yt=np.concatenate((y[posindex[0][0:nindex]],y[negindex[0][0:nindex]]))
    xt=np.concatenate((x[posindex[0][0:nindex]],x[negindex[0][0:nindex]]))
    
    return xt,yt

def balanceup(x,y):
    posindex=np.where( y == 1 )
    negindex=np.where( y == 0 )
    xt=[]
    yt=[]
    yindex=[]
    
    if(len(posindex[0])!=0 and len(negindex[0])!=0):
       
        nindex=max(len(posindex[0]),len(negindex[0]))
        mini=min(len(posindex[0]),len(negindex[0]))
        diff=nindex-mini
        u=print(len(x[0]))
        for i in range(0,mini):
            yt.append(y[posindex[0][i]])
            yt.append(y[negindex[0][i]])
            xt.append(x[posindex[0][i]])
            xt.append(x[negindex[0][i]])
        #print('first',sum(yt)/len(yt)) 
        if(len(posindex[0])>len(negindex[0])):
            toextract=negindex
            enter=posindex
        else:
            toextract=posindex
            enter=negindex
        if(diff!=0 and len(toextract[0])!=0):
            for i in range(0,diff):
                r=np.random.randint(0,len(toextract))
                yt.append(y[toextract[0][r]])
                xt.append(x[toextract[0][r]])
                yt.append(y[enter[0][mini+i]])
                xt.append(x[enter[0][mini+i]])
    else:
        #print('Unbalance')
        u=1
        xt=x
        yt=y
    #print(sum(yt)/len(yt))              
    return xt,yt,u      


import warnings
from sklearn.ensemble import RandomForestClassifier
def binaryFeatureSelection(ind,window,sentiment,prices):
    warnings.filterwarnings('ignore')
    o=0
    accstocksresults=[]
    f1stocksresults=[]
    f1scoreresults=[]
    aucresults=[]
    chartlist=[]
    gainlist=[]


    for price in prices:
        #print('New price[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]')

       
        
        print('Working on...',files[o])

        senttemp=sentiment[o]
        senttemp=np.nan_to_num(np.asarray(senttemp, dtype=float))

        xtemp=price
        o+=1
        xtemp=np.nan_to_num(np.asarray(xtemp, dtype=float))
        xtemp=xtemp[:,ind]

        #print(xtemp)
        accintime=[]
        trendwindowtime=[]
        trendwindowtime.append(window)
        f1total=[]
        acctotal=[]
        auctotal=[]
        percfinal=[]
        f1scoretotal=[]
        percpostotal=[]
        percnegtotal=[]
        for t in trendwindowtime:
            #1
        #label because of the maket and append values without data
        #simo theroy past trend

            x=[]
            y=[]
            percentage=[]

            yvolatility=[]
            #print('============================================================')
            #print('Working on window:',t)
            #print(len(xtemp))
            ##QUI C E L'UNICO APPUNTO GUARDA SE CON +1 CAMBIA

            for i in range(0,len(price)-t-1):
                s=np.sign(price.iloc[i+t+1]['close']-price.iloc[i+1]['open'])
                percentage.append((100*(price.iloc[i+t+1]['close']-price.iloc[i+1]['open']))/price.iloc[i+1]['open']) 
                if(s==-1):
                    y.append(0)
                else:
                    y.append(1)
                yvolatility.append((100*abs(price.iloc[i+t+1]['close']-price.iloc[i+1]['open']))/price.iloc[i+1]['open'])

                x.append(np.concatenate((senttemp[i],xtemp[i])))

            y=np.array(y)
            x=np.array(x)
            x=normalize(x,axis=0,norm='max')
            

            percentage=np.array(percentage)
            permindex=range(0,len(x))
            #permindex=np.random.permutation(permindex)
            train=0.8
            nt=math.ceil(len(x)*train)
            trainvalindex=permindex[0:nt]
            testindex=permindex[nt:]

            yvolatility=np.array(yvolatility)
            x_tv=[]
            y_tv=[]
            x_test=[]
            y_test=[]
            x_tv=x[trainvalindex]
            y_tv=y[trainvalindex]
            x_test=x[testindex]
            y_test=y[testindex]
            yvolatilitytest=yvolatility[testindex]
            #create structure for percentile valuation
            distribution=[]
            
            step=(max(yvolatilitytest)-min(yvolatilitytest))/5
            for v in range(0,5):
                pindexes=[]
                for r in range(0,len(y_test)):
                    if(yvolatilitytest[r]<min(yvolatilitytest)+step+step*v and yvolatilitytest[r]>min(yvolatilitytest)+step*v):
                        pindexes.append(r)
                distribution.append(pindexes)


            cspace=np.logspace(-4,4,10)
            gspace=np.logspace(-4,4,10)
            bestsvm=None
            maxacc=0
            cvacc=0
            maxg=0
            maxc=0
            #print('Model Selection...')
            #model selection
            cvacc=0
            totu=0
            for c in cspace:
                #print()
                for g in gspace:

                    cvacclist=[]
                    #faccio cross validation
                    #start with 40% as train and 10% for validation and then i move in percentege
                    # 0-40 40-50
                    # 0-50 50-60
                    # 0-60 60-70 
                    #etc test set is completely external i do in some way error extimation changin the ticker
                    trainpoint=math.floor(len(x_tv)*0.40)
                    dimval=math.floor(trainpoint*0.25)
                    endval=trainpoint+dimval

                    for i in range(0,6):
                        #print('-----')
                        x_train=x_tv[0:trainpoint]
                        y_train=y_tv[0:trainpoint]
                        x_val=x_tv[trainpoint:endval]
                        y_val=y_tv[trainpoint:endval]
                        #print(trainpoint)
                        #print(endval)
                        #print(len(x_tv))
                        trainpoint=trainpoint+dimval
                        endval=endval+dimval
                        p=sum(y_train)/(len(y_train)-sum(y_train))
                        rbf_svm=svm.SVC(kernel='rbf',C=c,gamma=g)
                        x_train,y_train=smote(x_train,y_train)
                        rbf_svm.fit(x_train,y_train)


                        if(sum(y_val)+6<len(x_val) and sum(y_val)>6):
                            x_val,y_val=smote(x_val,y_val)
                        else:
                            totu=totu+1
                        prediction=rbf_svm.predict(x_val)

                        cvacclist.append(geometric_mean_score(y_val, prediction))

                    cvacc=sum(cvacclist)/len(cvacclist)
                    if(cvacc>maxacc):
                        #print(cvacc)
                        maxacc=cvacc
                        maxg=g
                        maxc=c
        
        accstocksresults.append(cvacc)

        
                        
           
    return sum(accstocksresults)/len(accstocksresults),ind

