import math
import numpy as np
np.random.seed(13)
def dataPreparation(price,t,xtemp):
    x=[]
    y=[]
    for i in range(0, len(price) - t-1):
        y.append(np.sign(price.iloc[i + t+1]['close'] - price.iloc[i+1]['open']))
        x.append(xtemp[i])
    y = np.array(y)
    x = np.array(x)
    # Split betwwen train-validation and test
    train = 0.8
    nt = math.ceil(len(x) * train)
    x_tv = []
    y_tv = []
    x_test = []
    y_test = []
    x_tv = x[:nt]
    y_tv = y[:nt]
    x_test = x[nt:]
    y_test = y[nt:]
    # Fairly sampling the tv
    posindex = np.where(y_tv > 0)
    negindex = np.where(y_tv < 0)

    yindex = []
    nindex = min(len(posindex[0]), len(negindex[0]))

    # for i in range(1,nindex):
    y_tvnew = np.concatenate((y_tv[posindex[0][0:nindex]], y_tv[negindex[0][0:nindex]]))
    x_tvnew = np.concatenate((x_tv[posindex[0][0:nindex]], x_tv[negindex[0][0:nindex]]))

    # Fairly sampling the test 50% 50%
    posindex = np.where(y_test > 0)
    negindex = np.where(y_test < 0)

    yindex = []
    nindex = min(len(posindex[0]), len(negindex[0]))

    # for i in range(1,nindex):
    y_testnew = np.concatenate((y_test[posindex[0][0:nindex]], y_test[negindex[0][0:nindex]]))
    x_testnew = np.concatenate((x_test[posindex[0][0:nindex]], x_test[negindex[0][0:nindex]]))


    return x_tvnew,y_tvnew,x_testnew, y_testnew