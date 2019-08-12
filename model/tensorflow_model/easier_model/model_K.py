import pandas as pd
import numpy as np
import tensorflow as tf
from load_data import Data
import matplotlib.pyplot as plt
import math

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import accuracy_score

news_per_hour = 10
n_y = 1 #Numero di output, Per ora sali / scendi poi metteremo neutrale

#hidden LSTM units
num_lstm_units = 100

num_lstm_layers = 1
#learning rate for adam
# learning_rate=0.2
learning_rate=0.0001

#size of batch 	
batch_size=128

class MyModel():

	def set_seed(seed=7):
		numpy.random.seed(seed)


	def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
	          num_epochs = 15, minibatch_size = 128,
	          num_lstm_units = 100, num_lstm_layers=1, set_verbosity=True):

		model = Sequential()
		for i in range(0,num_lstm_layers-1):
			model.add(LSTM(num_lstm_units,  activation='relu',  return_sequences=True, input_shape=(len(X_train[0]),len(X_train[0][0]))))

		model.add(LSTM(num_lstm_units, activation='relu', input_shape=(len(X_train[0]),len(X_train[0][0]))))
		model.add(Dense(1, activation='sigmoid'))

		opt=keras.optimizers.Adam(lr=learning_rate)

		# opt=keras.optimizers.SGD(lr=learning_rate)
		model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
		if(set_verbosity):
			print('Optimizer: Adam \tLearning Rate:'+str(learning_rate))
			model.summary()


		# shuffle: shuffling the Training data, in this way I think I can shuffle the training
		# history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=minibatch_size, shuffle=False, validation_split=0.2, verbose=1)
		history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=minibatch_size, validation_split=0.2, verbose=1)
		# summarize history for accuracy
		plt.figure(figsize=(20,10))
		plt.plot(history.history['acc'], 'b', label='accuracy_train')
		plt.plot(history.history['val_acc'], 'r', label='accuracy_validation')
		plt.plot(history.history['loss'], '--b', label='cost_train')
		plt.plot(history.history['val_loss'], '--r', label='cost_validation')
		plt.title("Learning rate =" + str(learning_rate))
		plt.xlabel('epoch')
		plt.legend()
		plt.plot()



		yhat = model.predict(X_test, batch_size=minibatch_size, verbose=1)
		yhat = [1 if x>0.5 else -1 for x in yhat]
		print('Test accuracy: '+str(accuracy_score(Y_test, yhat)))








#  MAIN
# .
# .
# .
# .
# .
# .
# .
# .

Data.load_data(momentum_window=30, X_window_average=30, newsTimeToMarket = 0)

(X_train, Y_train), (X_test, Y_test) = Data.get_train_test_set()

test_x = tf.convert_to_tensor(X_test, dtype=tf.float32)
train_x = tf.convert_to_tensor(X_train, dtype=tf.float32)

train_y = tf.convert_to_tensor(Y_train, dtype=tf.float32)
test_y = tf.convert_to_tensor(Y_test, dtype=tf.float32)

print('.........................')
print ("number of training examples = " + str(train_x.shape[0]))
print ("number of test examples = " + str(test_x.shape[0]))
print ("X_train shape: " + str(train_x.shape))
print ("Y_train shape: " + str(train_y.shape))
print ("X_test shape: " + str(test_x.shape))
print ("Y_test shape: " + str(test_y.shape))
print('.........................')



print ("Test baseline: " + str( np.sum(Y_test==1)/Y_test.shape[0]))
MyModel.model(X_train, Y_train, X_test, Y_test,  minibatch_size = batch_size, learning_rate = learning_rate,num_lstm_layers=num_lstm_layers, num_lstm_units = num_lstm_units)
