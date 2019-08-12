from __future__ import division

import numpy as np
#np.random.seed(19)
import keras
import load_data3
from keras.models import *
from keras import regularizers
from keras.layers import Input, Dense, Dot, merge, Flatten, Concatenate
import tensorflow as tf
from keras.preprocessing import sequence
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, merge
from keras.layers import LSTM, Lambda, Reshape, Dropout

import predictonlyup
import setBatchSize

#np.random.seed(1337)  # for reproducibility
skip_vector_dim = 2400
attention_input_dim = 10
# timeSpan_vector_dim = 2400 = skip_vector_dim sicuro per l'attention model



def build_model(lstmunits,regularization,batchsize,drop):
	# ATTENTION MODEL
	inputs = Input(batch_shape=(batchsize, attention_input_dim, skip_vector_dim))
	attention_alpha = Dense(1 ,activity_regularizer=regularizers.l2(regularization), activation='softmax', name='attention_vec')(inputs)
	attention = Dot(1)([attention_alpha,inputs])
	attention=Lambda(lambda y: keras.backend.permute_dimensions(y,(1,0,2)))(attention)
	attention=Dropout(drop)(attention)
	digits=LSTM(lstmunits,return_sequences=True,stateful=True)(attention)
	digits=Dropout(drop)(digits)

	#digits=Lambda(lambda y: keras.backend.permute_dimensions(y,(1,0,2)))(digits)
	digits = Lambda(lambda y: keras.backend.squeeze(y, 0))(digits)
	classification=Dense(1, activation='sigmoid',activity_regularizer=regularizers.l2(regularization))(digits)
	model = Model(input=[inputs], output=classification)

	return model

def createDataset(data,label, time_steps):
	data_x, data_y = [], []
	for i in range(len(data) - time_steps - 1):
		a = data[i:(i + time_steps)]
		data_x.append(a)
	for i in range(len(label)-time_steps-1):
		data_y.append(label[i + time_steps])
	return np.array(data_x), np.array(data_y)


def findMaxAcc(valacc,valloss):
	prevloss=valloss[0]
	for i in range(1,len(valloss)):
		if(valloss[i]>prevloss):
			return valacc[i-1]
	print('Problems with find MaxAccuracy')
	return valacc[len(valacc)-1]

def model(learning,lstmunits,regularization,drop,x_train, y_train, x_test, y_test):
	max_features = 2400
	batch_size = 10
	#print('============================================================')
	x_train, y_train, x_test, y_test = setBatchSize.setBatchSize(x_train, y_train, x_test, y_test, batch_size)
	#print('============================================================')
	x_train = np.nan_to_num(np.asarray(x_train, dtype=float))
	y_train = np.nan_to_num(np.asarray(y_train, dtype=float))
	x_test = np.nan_to_num(np.asarray(x_test, dtype=float))
	y_test = np.nan_to_num(np.asarray(y_test, dtype=float))

	y_test = (y_test + 1) / 2
	y_train = (y_train + 1) / 2

	model = build_model(drop=drop,lstmunits=lstmunits,regularization=regularization,batchsize=batch_size)
	opt=keras.optimizers.Adam(lr=learning)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	print(model.summary())
	#print('Train...')
	uptr = predictonlyup.alwaysUp(y_train)
	upts = predictonlyup.alwaysUp(y_test)
	#print('Only up:', uptr)
	#print('Only up:', upts)

	history=model.fit(x_train, y_train, batch_size=batch_size, epochs=20,verbose=0, shuffle=False, validation_data=(x_test, y_test))
	maxAcc=findMaxAcc(history.history['val_acc'],history.history['val_loss'])

	#print('Here you are the max acc')
	#print(maxAcc)
	'''''
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy :')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss :')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	#score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
	#print('Test score:', score)
	#print('Test accuracy:', acc)
	'''
	return maxAcc, upts


