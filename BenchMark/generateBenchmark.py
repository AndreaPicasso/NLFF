import pandas as pd
import re
import naiveBayes
from os import listdir
import numpy as np
from stockstats import StockDataFrame
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data

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

# PARAMETRO PER SETTARE LA MOVING STD DEVIATION
param = "3"

#std = stock.get('close_' + param + '_mstd')
# print(std)
toSave = pd.DataFrame(stock)
label = list()
dellabel=list()
for i in range(0, len(toSave)):
    value = 0
    dif = toSave['close'][i] - toSave['open'][i]
    if (dif > 0):
        # positivo
        value = 1
    if (dif <= 0):
        value = 0
    label.append(value)

up=naiveBayes.alwaysUp(label)
print('Predict only up:')
print(up)
alphas=np.linspace(0.1,1,9)
splits=np.linspace(0.1,0.9,8)
maxacc=0
maxalpha=0
maxsplit=0
print('Naive Bayes Gaussian:')

for split in splits:

    accuracy=naiveBayes.naiveBayesGaussian(label[0:len(label)-2],label[1:len(label)-1], split)
    if(accuracy>maxacc):
        maxacc=accuracy
        maxsplit=split

print(maxacc,maxsplit)

print('Naive Bayes Bernoulli:')
for alpha in alphas:
    for split in splits:

        accuracy=naiveBayes.naiveBayesBernoulli(label[0:len(label)-2],label[1:len(label)-1], split,alpha)
        if(accuracy>maxacc):
            maxacc=accuracy
            maxalpha=alpha
            maxsplit=split

print(maxacc, maxalpha,maxsplit)

