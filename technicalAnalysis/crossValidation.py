#for obtaining all traindata simply split the dataset with load data with 70%for train(crossval) and 30 for test
from __future__ import division
import modelStateful
import math
def crossValidation(x_data, y_data, nfold,learning,mem,regu,drop):
    dimfolf=(len(x_data)//nfold)
    # -1 so the last fold is bigger ex 100/8=12,5 last fold from 11 to 12,5
    acclist=list()
    baselist=list()
    first=math.floor(nfold*0.7)
    for i in range(first,nfold):
        #print('Fold:',i)
        if(i==nfold-1):
            xdataToPass=x_data
            ydataToPass = y_data
            acc, base = execModel(xdataToPass, ydataToPass, learning, mem, regu,drop)
            acclist.append(acc)
            baselist.append(base)
        else:
            xdataToPass=x_data[0:dimfolf*i]
            ydataToPass = y_data[0:dimfolf * i]
            acc,base=execModel(xdataToPass, ydataToPass,learning,mem,regu,drop)
            acclist.append(acc)
            baselist.append(base)

    return acclist,baselist

def splitData(x_data,y_data, valperc):
    val=math.ceil(len(x_data)*valperc)
    x_train=x_data[0:val]
    y_train=y_data[0:val]
    x_val=x_data[val:]
    y_val=y_data[val:]
    return x_train,y_train,x_val,y_val


def execModel(x_data,y_data,learning,mem,regu,drop):
    x_train,y_train,x_val,y_val=splitData(x_data,y_data,0.7)
    maxAcc,base=modelStateful.model(learning, mem, regu,drop, x_train, y_train, x_val, y_val)
    return maxAcc,base