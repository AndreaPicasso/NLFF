import pandas as pd
import numpy as np


class Indicators:

	def momentum(prices, window):
	    momentum_indicator = (prices['close'] / prices['close'].shift(window))-1
	    return momentum_indicator

	def SMA(prices, window):
	    SMA_indicator = (prices['close'] / prices['close'].rolling(window).mean())-1
	    return SMA_indicator
	    
	def inBBands(prices):
	    inBB = []
	    inBB.append(0)
	    direction = 0   
	    for i in range(1,len(prices)):
	        if(prices['close'][i] > prices['boll_ub'][i] and prices['close'][i-1] < prices['boll_ub'][i-1]):
	            direction = 0
	        if(prices['close'][i] < prices['boll_ub'][i] and prices['close'][i-1] > prices['boll_ub'][i-1]):
	            direction = -1
	        if(prices['close'][i] > prices['boll_lb'][i] and prices['close'][i-1] < prices['boll_lb'][i-1]):
	            direction = 1
	        if(prices['close'][i] < prices['boll_lb'][i] and prices['close'][i-1] > prices['boll_lb'][i-1]):
	            direction = 0
	        inBB.append(direction)
	    inBB = np.convolve(inBB, np.repeat(1.0, 3)/3, 'same')
	    return inBB



	def eccessOfVolumes(price):
	    eccessOfVolumes = []
	    means = list()
	    runningMean = 0
	    for i in range(len(price['volume'])):
	        runningMean +=price['volume'][i]
	        if(i-500 >= 0):
	        	runningMean -=price['volume'][i-500]
	        means.append(runningMean/min(i+1, 500))
	    for i in range(len(price['volume'])):
	        eccessOfVolumes.append(price['volume'][i] - means[i])

	    eccessOfVolumes = np.convolve(eccessOfVolumes, np.repeat(1.0, 5)/5	, 'same')
	    return eccessOfVolumes

	def aaron(prices, window):
		aaronUp = [0]*window #Date of highest
		aaronDown = [0]*window #Date of lowest
		aaron = [0]*window

		for x in range(window, len(prices)):
			up = (window - np.argmax(np.asarray(prices['high'].iloc[x-window:x])))/window
			down = (window - np.argmin(np.asarray(prices['low'].iloc[x-window:x])))/window
			aaronUp.append(up)
			aaronDown.append(down)
			aaron.append(up-down)
			x +=1

		return aaron


	def MACD(prices, slowWindow=26, fastWindow=12):
		slow = prices['close'].ewm(com=slowWindow).mean()
		fast = prices['close'].ewm(com=fastWindow).mean()
		return fast - slow


	def RSI(prices, window):
		delta = prices['close'].diff()
		up = delta.copy()
		down = delta
		up[up<0] = 0
		down[down>0] = 0
		up = up.rolling(window).mean()
		down = down.rolling(window).mean().abs()
		RS = up/down
		return 1 - (1 / (1.0 + RS))