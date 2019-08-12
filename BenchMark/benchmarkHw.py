from __future__ import division

import HW
import pandas as pd
import matplotlib.pyplot as plt
import re

import naiveBayes
import load_data
from os import listdir
import numpy as np
from stockstats import StockDataFrame
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
import math
param = {
    'q': "AAPL",  # Stock symbol (ex: "AAPL")
    'i': "1200",  # Interval size in seconds ("86400" = 1 day intervals)
    'x': "NASD",  # INDEXNASDAQStock exchange symbol on which stock is traded (ex: "NASD")
    'p': "2Y"  # Period (Ex: "1Y" = 1 year)
}

# link to doc http://www.networkerror.org/component/content/article/1-technical-wootness/44-googles-undocumented-finance-api.html
# get price data (return pandas dataframe) 1465948800
priceData = get_price_data(param)
stock = StockDataFrame.retype(priceData)
toSave = pd.DataFrame(stock)
print('Sample data only for seeing if stockstat is working')
print(toSave.head())
#parameter for train vs test
percentage=0.9
firsttrain=math.floor(len(toSave)*percentage)
test=2
##Train on all the dataset incremental from percentage selected with only one test incremental test is 2 because need to make delta price
print('WORKING.....')
data = toSave['close'].values.tolist()
accuracy=0

for k in range(firsttrain,len(toSave)-2):
    print('Train dim:',k)
    print('Dataset dim:',len(toSave)-2)
    train=k
    realdata=data[0:k+1]
    forecast, alpha, beta, gamma, rmse=HW.multiplicative(realdata, train, test, alpha = None, beta = None, gamma = None)


    df=forecast[1]-forecast[0]
    dd=data[train+1]-data[train]
    prod=df*dd
    if(prod>0):
        accuracy+=1

    print('Print of accuracy only for seeing if all is going well',accuracy)

    print('Finished with one train')
accuracy=accuracy/(len(toSave)-firsttrain)
print('Final accuracy=:',accuracy)