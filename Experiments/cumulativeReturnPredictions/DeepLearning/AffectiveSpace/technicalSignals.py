import pandas as pd
import numpy as np



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