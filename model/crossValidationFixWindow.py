#for obtaining all traindata simply split the dataset with load data with 70%for train(crossval) and 30 for test
from __future__ import division
import modelStateful
import math
def crossValidation(x_data, y_data, nfold,learning,mem,regu,drop,windowdim):
    dimfolf=(len(x_data)//nfold)
    # -1 so the last fold is bigger ex 100/8=12,5 last fold from 11 to 12,5
    acclist=list()
    baselist=list()
    move=(len(x_data)-windowdim)//nfold
    for i in range(0,nfold):

        xdataToPass=x_data[0+move*i:windowdim+move*i]
        ydataToPass = y_data[0+move*i:windowdim+move*i]
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