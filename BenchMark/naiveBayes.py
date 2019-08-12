from __future__ import division
import sklearn
import math
import numpy as np
from sklearn.naive_bayes import BernoulliNB ,GaussianNB, MultinomialNB
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

def alwaysUp(data):
    #supposed to be -1 or 1
    #supposed to predict everytime 1 (up)
    data=np.asarray(data)
    up=(data == 1).sum()/len(data)
    return up
def naiveBayesBernoulli(data,labels, trainPercentage,alpha):
    #i'll have to split into train & test
    split=len(data)*trainPercentage
    split = math.floor(split)
    trainData=np.asarray(data[0:split]).reshape(-1, 1)
    testData=np.asarray(data[split:]).reshape(-1, 1)
    trainLabel=np.asarray(labels[0:split]).reshape(-1, 1)
    testLabel=np.asarray(labels[split:]).reshape(-1, 1)
    trainData,trainLabel=smote(trainData,trainLabel)
    gnb = BernoulliNB(alpha=alpha)
    y_pred = gnb.fit(trainData, trainLabel.ravel()).predict(testData)
    testLabel=testLabel.ravel()
    accuracy=((testLabel == y_pred).sum())/len(testLabel)
    return accuracy

def naiveBayesGaussian(data,labels, trainPercentage):
    #i'll have to split into train & test
    split=len(data)*trainPercentage
    split = math.floor(split)
    trainData=np.asarray(data[0:split]).reshape(-1, 1)
    testData=np.asarray(data[split:]).reshape(-1, 1)
    trainLabel=np.asarray(labels[0:split]).reshape(-1, 1)
    testLabel=np.asarray(labels[split:]).reshape(-1, 1)
    trainData,trainLabel=smote(trainData,trainLabel)
    gnb = GaussianNB()
    y_pred = gnb.fit(trainData, trainLabel.ravel()).predict(testData)
    testLabel=testLabel.ravel()
    accuracy=((testLabel == y_pred).sum())/len(testLabel)
    return accuracy

    return accuracy
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
        u=0
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