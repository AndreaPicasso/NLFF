from __future__ import division
import numpy as np
np.random.seed(13)
import predictonlyup

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
from keras.layers import LSTM, Lambda, Reshape



#np.random.seed(1337)  # for reproducibility
skip_vector_dim = 2400
attention_input_dim = 10
# timeSpan_vector_dim = 2400 = skip_vector_dim sicuro per l'attention model



def build_model(time_steps,lstmunits,regularization):
	# ATTENTION MODEL
	inputs = Input(shape=(time_steps,attention_input_dim, skip_vector_dim))
	attention_alpha = Dense(1 ,activity_regularizer=regularizers.l2(regularization),input_shape=( time_steps,skip_vector_dim, attention_input_dim), activation='softmax', name='attention_vec')(inputs)
	attention = Dot(2)([attention_alpha,inputs])
	attention=Lambda(lambda y: keras.backend.squeeze(y,2),output_shape=(time_steps,skip_vector_dim))(attention)
	digits=LSTM(lstmunits)(attention)
	classification=Dense(1, activation='sigmoid')(digits)
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




def model(learning,time_steps,lstmunits,regularization,x_train, y_train, x_test, y_test):
	max_features = 2400
	batch_size = 10
	x_train = np.nan_to_num(np.asarray(x_train, dtype=float))
	y_train = np.nan_to_num(np.asarray(y_train, dtype=float))
	x_test = np.nan_to_num(np.asarray(x_test, dtype=float))
	y_test = np.nan_to_num(np.asarray(y_test, dtype=float))
	#code for build the slices
	x_train, y_train = createDataset(x_train, y_train, time_steps)
	x_test, y_test = createDataset(x_test, y_test, time_steps)
	y_test = (y_test + 1) / 2
	y_train = (y_train + 1) / 2
	model = build_model(time_steps=time_steps,lstmunits=lstmunits,regularization=regularization)
	opt=keras.optimizers.Adam(lr=learning)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	print(model.summary())
	print('Train...')
	uptr=predictonlyup.alwaysUp(y_train)
	upts=predictonlyup.alwaysUp(y_test)
	print('Only up:', uptr)
	print('Only up:', upts)


	history=model.fit(x_train, y_train, batch_size=batch_size,epochs=30,verbose=1, validation_data=(x_test, y_test))
	print('Here you are history')

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


	score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)
	return score,acc



