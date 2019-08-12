from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from math import sqrt

def sign(x):
    if x >= 0:
        return 1.0
    elif x < 0:
        return -1.0
        #return 0



def predict(X,timeToMarket=0, order = (0,2), test_percentage = 0.3):
	idx_split = math.floor(len(X)*(1-test_percentage))
	y = X[idx_split:]
	yhat = list()
	for t in range(idx_split, len(X)):
		history=X[:t-1-timeToMarket-50] # 30 because we need to predict the future, otherwise y_t and y_t+1 are related
		model = ARMA(history, order = order)
		model_fit = model.fit( disp=0 ) #suppress info
		yhat.append(model_fit.forecast()[0])
	return (yhat, y)

	# idx_split = math.floor(len(X)*(1-test_percentage))
	# y = X[idx_split:]
	# history=X[:idx_split-timeToMarket]
	# model = ARMA(history, order = order)
	# model_fit = model.fit( disp=0 ) #suppress info
	# # one step forecast
	# yhat = model_fit.forecast(steps=len(y))[0]

	# # store forecast ob
	# return (yhat, y)





def cost(Y_hat, Y):
	assert len(Y_hat) == len(Y)
	# This is classification
	return np.sum(np.sign(Y_hat) != np.sign(Y)) / len(Y)

	# Use MLE instead:
	# return  ((np.array(Y_hat) - np.array(Y)) ** 2).mean()



def accuracy(Y_hat, Y):
	print(len(Y))
	print(np.sum(np.sign(Y_hat) == np.sign(Y)) )
	assert len(Y_hat) == len(Y)
	return np.sum(np.sign(Y_hat) == np.sign(Y)) / len(Y)



'''
 https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average

y(t) + sum(ai * y(t-i) ) = sum( bj * e(t-j) )

e(t): error at time t (assumed to be a gaussian with mean = 0)


ARIMA(p,d,q) where parameters p, d, and q are non-negative integers:
p is the order (number of time lags) of the autoregressive model 								-> len(ai)
d is the degree of differencing (the number of times the data have had past values subtracted)	-> d=1 implies instead of y(t) we consider d y(t) / dt
q is the order of the moving-average model.														-> len(bj)


ADF test: tests the null hypothesis that a unit root is present in a time series sample



UNIT ROOT: a unit root is a feature of some stochastic processes that can cause problems in statistical inference involving time series models
If the other roots of the characteristic equation lie inside the unit circle—that is, have a modulus (absolute value) less than one—then the first difference of the process will be stationary;
otherwise, the process will need to be differenced multiple times to become stationary
example: suppose y(t) = a1 y(t-1) + e(t)
if a1 == 1 the process has a unit root -> moments (mean, variance, ...) of the stochastic process depend on t
infact: y(t) = y(0) + sum(e(t))    ---> variance(y(t)) = t * sigma    (where variance(y(1)) = sigma )




Stationary process: is a stochastic process whose unconditional joint probability distribution does not change when shifted in time.
Consequently, parameters such as mean and variance, if they are present, also do not change over time



'''

tickers = ['AAPL','AMZN','GOOGL','MSFT','FB','INTC','CSCO','CMCSA','NVDA','NFLX']       
for tic in tickers:
	print('\n\n\n==================== '+str(tic)+' ==================== \n\n\n')
	momentum_window = 30
	prices = pd.read_csv('/home/simone/Desktop/NLFF/indexes/indexes'+str(tic)+'.csv')
	z = list()
	for i in range(0,momentum_window):
	    z.append(523) #Valore impossibile per fare drop successivamente  
	for i in range(momentum_window,prices.shape[0]):
	    z.append(sign(prices['close'][i] - prices['close'][i-momentum_window]))
	prices['close'] = z

	prices = prices[prices['close'] != 523]
	prices = prices['close'].as_matrix()
	prices = np.squeeze(prices)
	# Calcola differenza dei prezzi: out[n] = a[n+1] - a[n]
	# otherwise we could just put d =1 into the ARIMA model

	# l-ultimo e scartato: len(np.diff(a)) == len(a)-1
	# prices = np.diff(prices)
	Ttm_range = [0, 7, 14, 21, 35, 70, 105, 210]
	MCCs = list()
	accs = list()
	for ttm in Ttm_range:
		print(ttm)
		(yhat, y) = predict(prices, timeToMarket=ttm)
		accs.append(accuracy(yhat, y))

		cm = confusion_matrix(np.sign(y), np.sign(yhat))
		tn, fp, fn, tp = cm.ravel()
		if(tp + fp == 0):
		    tp = 1
		if(tp + fn == 0):
		    tp = 1
		if(tn + fp == 0):
		    tn = 1
		if(tn + fn == 0):
		    tn = 1
		MCCs.append((tp*tn -fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))


	print(Ttm_range)
	print(accs)
	print(MCCs)



	# bestP = 0
	# bestD = 0
	# bestQ = 0
	# bestError = np.inf
	# bestPrediction = []

	# for p in range(0, 2):
	# 	for q in range(0, 4):
	# 		print('\n('+str(p)+','+str(d)+','+str(q)+')')
	# 		try:
	# 			(yhat, y) = predict(prices, order = (p,q))
	# 			# in caso di eccezione l'accuracy sara la stessa di prima e il best non verra aggiornato0
	# 			# currentError = accuracy(predictions, prices[0:-34])
	# 			currentError = cost(yhat, y)

	# 			if(currentError < bestError):
	# 				bestP = p
	# 				bestD = d
	# 				bestQ = q
	# 				bestError = currentError
	# 				bestPrediction = yhat
	# 			#print('Accuracy: '+str(accuracy(yhat, y)))
	# 		except Exception as e:
	# 			print('------------  ERROR: '+str(e))
	# 		#print('BestError ('+str(bestP)+','+str(bestD)+','+str(bestQ)+'):'+ str(bestError)+ ' CurrentError: '+ str(currentError))	





