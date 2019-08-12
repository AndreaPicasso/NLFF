#for obtaining all traindata simply split the dataset with load data with 70%for train(crossval) and 30 for test
from __future__ import division
import modelStateful
import itertools
import math
import numpy as np
def crossValidation(x_data, y_data, nfold,learning,mem,regu,drop):
    dimfolf=(len(x_data)//nfold)
    # -1 so the last fold is bigger ex 100/8=12,5 last fold from 11 to 12,5
    acclist=list()
    baselist=list()

    for i in range(0,nfold):

        #print('Fold:',i)
        xdataToPass=x_data
        ydataToPass = y_data
        acc, base = execModel(xdataToPass, ydataToPass, learning, mem, regu,drop,i,dimfolf)
        acclist.append(acc)
        baselist.append(base)


    return acclist,baselist

def splitData(x_data,y_data, i,dimfold):
    print('0')
    print(i*dimfold)
    print((i+1)*dimfold)
    print('end')
    print('---')
    print(dimfold)
    x_train=np.empty((1))
    y_train=np.empty((1))
    x_val=np.empty((1))
    y_val=np.empty((1))

    for k in range(0,len(x_data)-1):
        if(k>i*dimfold and k<(i+1)*dimfold):
            x_val=np.append(x_val,x_data[k])
            y_val = np.append(y_val, y_data[k])
        else:
            x_train=np.append(x_train,x_data[k])
            y_train = np.append(y_train, y_data[k])
    #x_train=np.append(np.asarray(x_data[0:i*dimfold]),np.asarray(x_data[(i+1)*dimfold:]))


    print(len(x_train))

    print(len(y_train))

    print(len(x_val))

    print(len(y_val))
    return x_train,y_train,x_val,y_val


def execModel(x_data,y_data,learning,mem,regu,drop,i,dimfold):
    x_train,y_train,x_val,y_val=splitData(x_data,y_data,i,dimfold)
    maxAcc,base=modelStateful.model(learning, mem, regu,drop, x_train, y_train, x_val, y_val)
    return maxAcc,base