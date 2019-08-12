import pandas as pd
import talib
import re
from os import listdir
import numpy as np
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data
files=list()
files=listdir('/home/andrea/Desktop/NLFF/intrinioDataset')
print(files)
#for t in lista:
 #   t=re.sub('\.csv$', '', t)
  #  print(t)
files=['FB']
#files=["AAL","FB"]
for file in files:
    print("Working on...")
    print(file)
    file=re.sub('\.csv$', '', file)
    print(file)

    param = {
    'q': file, # Stock symbol (ex: "AAPL")
    'i': "3600", # Interval size in seconds ("86400" = 1 day intervals)
    'x': "NASD", # INDEXNASDAQStock exchange symbol on which stock is traded (ex: "NASD")
    'p': "1Y" # Period (Ex: "1Y" = 1 year)
    }
   
    # link to doc http://www.networkerror.org/component/content/article/1-technical-wootness/44-googles-undocumented-finance-api.html
    # get price data (return pandas dataframe) 1465948800
    priceData = get_price_data(param)
    priceData.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    print(priceData['Close'].values)
    close=np.transpose(priceData['Close'].values)
    print(close)
    #indexes=np.array(priceData)
    #print(indexes)
    #print(priceData.head())
    indicators=list()
    indicators.append(talib.MACD(priceData['Close']))

    #indicators.append(talib.MOM(priceData['Close']))
    #indicators.append(talib.RSI(priceData['Close']))
    #indicators.append(talib.EMA(priceData['Close']))
    #indicators.append(talib.BBANDS(priceData['Close']))
    #print(pd.DataFrame(indicators).head())
    #toSave = pd.concat([priceData, pd.DataFrame(indicators)], axis=1)
    #print(toSave)
        # print(toSave)
        # dat1 = pd.DataFrame({'dat1': [9,5]})
        # toSave.join(pd.DataFrame({x:stock.get(x)}))
        # print("date"+toSave)
    if (len(indicators) > 0):
        stringa='/home/andrea/Desktop/NLFF/DatasetIndexesTalib/indexes'+str(file)+'.csv'
        pd.DataFrame(indicators).to_csv(stringa)
    else:
        print("Non trovato " + file)
    