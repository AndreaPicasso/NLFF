from __future__ import division

import numpy as np
import keras
import load_data
from keras.models import *
from keras.layers import Input, Dense, Dot, merge
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, merge
from keras.layers import LSTM, Lambda




(x_train, y_train), (x_test, y_test) =load_data.load_data(0.3)
print(len(np.argwhere(np.isnan(np.asarray(x_train,dtype=float)))))
x_train=np.nan_to_num(np.asarray(x_train,dtype=float))
y_train=np.nan_to_num(np.asarray(y_train,dtype=float))
x_test=np.nan_to_num(np.asarray(x_test,dtype=float))
y_test=np.nan_to_num(np.asarray(y_test,dtype=float))
print('--------------------------------------------------------------------')
print(np.argwhere(np.isnan(x_train)))
print('--------------------------------------------------------------------')

pos=[]
neg=[]
pos=x_train[np.argwhere(y_train==1)]
neg=x_train[np.argwhere(y_train==-1)]
distpos=0
dist=0
distneg=0
for i in range(0,len(x_train)-1):
    dist+=np.linalg.norm(x_train[i]-x_train[i+1])
dist=dist/len(x_train)
for i in range(0,len(pos)-1):
    distpos += np.linalg.norm(pos[i]-pos[i+1])
distpos=distpos/len(pos)
for i in range(0,len(neg)-1):
    distneg += np.linalg.norm(neg[i]-neg[i+1])
distneg=distneg/len(neg)
print('dist', dist)
print('distpos', distpos)
print('distneg', distneg)

