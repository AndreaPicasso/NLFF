import pandas as pd
import re
import naiveBayes
from os import listdir
import numpy as np
from stockstats import StockDataFrame
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data

param = {
    'q': "AAPL",  # Stock symbol (ex: "AAPL")
    'i': "3600",  # Interval size in seconds ("86400" = 1 day intervals)
    'x': "NASD",  # INDEXNASDAQStock exchange symbol on which stock is traded (ex: "NASD")
    'p': "1Y"  # Period (Ex: "1Y" = 1 year)
}

# link to doc http://www.networkerror.org/component/content/article/1-technical-wootness/44-googles-undocumented-finance-api.html
# get price data (return pandas dataframe) 1465948800
priceData = get_price_data(param)
# print(priceData)
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
        value = -1
    label.append(value)


up=naiveBayes.alwaysUp(label)
print('Predict only up:')
print(up)
print('Simple naivebayes:')
accuracy=naiveBayes.naiveBayesGaussian(label[:len(label)-2],label[1:len(label)-1], 0.7)
print(accuracy)