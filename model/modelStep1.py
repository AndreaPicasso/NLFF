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



#np.random.seed(1337)  # for reproducibility
skip_vector_dim = 2400
attention_input_dim = 10
# timeSpan_vector_dim = 2400 = skip_vector_dim sicuro per l'attention model



def build_model():
	# ATTENTION MODEL
	inputs = Input(shape=(attention_input_dim, skip_vector_dim))
	attention_alpha = Dense(1, input_shape=( skip_vector_dim, attention_input_dim), activation='softmax', name='attention_vec')(inputs)
	attention = Dot(1)([attention_alpha,inputs])
	digits=LSTM(10)(attention)
	print(keras.backend.shape(digits))
	classification=Dense(1, activation='sigmoid')(digits)
	model = Model(input=[inputs], output=classification)
	print(model.summary())
	return model







max_features = 2400
batch_size = 10
# 70000 sample dim 6x2400 poiche 6 embeddings
#x_train = np.random.random((700,6,2400))#np.ones((70000,6, 2400))
#x_test = np.random.random((700,6,2400))#np.ones((70000,6, 2400))
#y_test =  np.random.randint(2, size=(700))#np.ones(70000)
#y_train = np.random.randint(2, size=(700))#np.ones(70000)

(x_train, y_train), (x_test, y_test) =load_data.load_data(0.3)
print(len(np.argwhere(np.isnan(np.asarray(x_train,dtype=float)))))
x_train=np.nan_to_num(np.asarray(x_train,dtype=float))
y_train=np.nan_to_num(np.asarray(y_train,dtype=float))
x_test=np.nan_to_num(np.asarray(x_test,dtype=float))
y_test=np.nan_to_num(np.asarray(y_test,dtype=float))
print('--------------------------------------------------------------------')
print(np.argwhere(np.isnan(x_train)))
print('--------------------------------------------------------------------')

y_test=(y_test+1)/2
y_train=(y_train+1)/2
print(x_train)
model = build_model()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Train...')
model.fit(x_train, y_train,batch_size=batch_size,epochs=10,verbose=1,validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)