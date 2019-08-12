import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from load_data import Data
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
#tf.logging.set_verbosity(tf.logging.INFO)






#  MODEL from SKIP THROUGH onwards
# 
# 


news_per_hour = 10
skip_vector_dim = 2400
n_y = 1 #Numero di output, Per ora sali / scendi poi metteremo neutrale


# Conta fino a 100 ore prima --------------------------------------- VEDI BENE
time_steps=100
#hidden LSTM units
num_lstm_units=100

#learning rate for adam
learning_rate=0.001
#size of batch 	
batch_size=128






class MyModel():

	def create_placeholders():
	    X = tf.placeholder(tf.float32, shape=(None, news_per_hour, skip_vector_dim), name='X')
	    Y = tf.placeholder(tf.float32, shape=(None, n_y), name='Y')
	    return X, Y



	def forward_propagation(X):
   

	    # ATTENTION
		e = tf.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
		alpha = tf.nn.softmax(e, name='attention_weights')   												# tf.nn.softmax(logits,axis=None, ..)
		timeSlotEmbeddings =  tf.matmul(alpha, X, transpose_a=True, name='timeSlotEmbeddings')				# tf.matmul(a,b, transpose_a=False, transpose_b=False, name=None )

		timeSlotEmbeddings = tf.squeeze(timeSlotEmbeddings, axis=1)											# timeSlotEmbeddings.shape = (?, 1, 2400) -> we need to convert it to (?, 2400)

		# # LSTM
		# # (see https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)


		# #Creo la sequenza di tensor da fornire all'lstm: un tensor [batch_size,input_size] per ogni step dell'lstm
		# # cioe il sample sara costituito da x_{i- time_steps} , ...,  x_{i-1}, x_{i}

		# NON HO FATTO CONTROLLI -----------
		lenghtTime = tf.shape(timeSlotEmbeddings)[0]
		timeSlotEmbeddings = tf.pad(timeSlotEmbeddings, [[time_steps-1,0],[0,0]])
		timeSlotSequence = list()
		for i in range(0, time_steps):
			timeSlotSequence.append(timeSlotEmbeddings[i:i+lenghtTime])			


		#reverse della lista
		#timeSlotSequence =  list(reversed(timeSlotSequence))


		lstm_layer=rnn.BasicLSTMCell(num_lstm_units)													# Definisco il layer
		outputs,_=rnn.static_rnn(lstm_layer, timeSlotSequence, dtype="float32")							# Definisco la rete ricorrente tramite il layer precedente
		'''
		STATIC RNN:

		  state = cell.zero_state(...)
		  outputs = []
		  for input_ in inputs:																			# timeSlotSequence dal piu recente al piu vecchio
		    output, state = cell(input_, state)
		    outputs.append(output)
		  return (outputs, state)

		BasicLSTMCell(inputs, state):
			Run this RNN cell on inputs, starting from the given state.
			inputs: 2-D tensor with shape [batch_size, input_size]

		
		'''
		prediction = tf.layers.dense(outputs[-1], 1, activation=tf.nn.sigmoid) 

		return prediction



	def compute_cost(Y_hat, Y):
	    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y))		# Y * -log(sigmoid(Y_hat)) + (1 - Y) * -log(1 - sigmoid(Y_hat))
	    #cost = tf.reduce_mean(tf.squared_difference(Y_hat, Y))
	    return cost


	def random_mini_batches(X_train, Y_train, minibatch_size):
		minibatches = list()

		m = int(len(X_train))
		minibatches.append((X_train[0:minibatch_size], Y_train[0:minibatch_size]))
		iterSize = minibatch_size
		while(iterSize < m):
			if(iterSize+minibatch_size < m):
				minibatches.append((X_train[iterSize:iterSize+minibatch_size], Y_train[iterSize:iterSize+minibatch_size]))
				iterSize += minibatch_size
			else:
				minibatches.append((X_train[iterSize:m],Y_train[iterSize:m]))
				iterSize = m
		return minibatches


	def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
	          num_epochs = 50, minibatch_size = 128,
	          num_lstm_units = 100, news_per_hour = 10, set_verbosity=True):
	    if(set_verbosity):
	    	print ("Learning rate: " +str(learning_rate))

	    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
	    m =int(len(X_train))
	    costs_train = []                                        # To keep track of the cost
	    costs_test = []                                        # To keep track of the cost
	    accuracy_train = []                                        # To keep track of the cost
	    accuracy_test = []                                        # To keep track of the cost

	    # Create Placeholders of the correct shape
	    X, Y = MyModel.create_placeholders()
	    
	    # Forward propagation: Build the forward propagation in the tensorflow graph
	    prediction = MyModel.forward_propagation(X)
	    
	    # Cost function: Add cost function to tensorflow graph
	    cost = MyModel.compute_cost(prediction, Y)
	    
	    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
	    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	    #grads = tf.train.AdamOptimizer(learning_rate = learning_rate).compute_gradients(cost)

	    # Initialize all the variables globally
	    init = tf.global_variables_initializer()

	    #This is for computing the test accuracy every epoch
	    predict_op = tf.to_float(prediction > 0.5)
	    correct_prediction = tf.equal(predict_op, Y)
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	     
	    # Start the session to compute the tensorflow graph
	    with tf.Session() as sess:

	        # Run the initialization
	        if(set_verbosity):
	        	print('initialization')
	        sess.run(init)
	        # Do the training loop
	        for epoch in range(num_epochs):
	            minibatch_cost = 0.
	            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
	            if(num_minibatches == 0):
	            	num_minibatches = 1
	            minibatches = MyModel.random_mini_batches(X_train, Y_train, minibatch_size)

	            for minibatch in minibatches:
	                # Select a minibatch
	                (minibatch_X, minibatch_Y) = minibatch
	                minibatch_X = np.asarray(minibatch_X)
	                minibatch_Y = np.asarray(minibatch_Y).reshape((len(minibatch_Y), 1))

	                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
	                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

	                # weights = tf.get_default_graph().get_tensor_by_name('dense_1/kernel:0')
	                # print(sess.run(weights))

	                minibatch_cost += temp_cost / num_minibatches
	                
	            # Print the cost every epoch

	            if  epoch % 1 == 0:
	                trainCost = sess.run(cost, feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1))})
	                costs_train.append(trainCost)
	                testCost = sess.run(cost, feed_dict={X: np.asarray(X_test), Y: np.asarray(Y_test).reshape((len(Y_test), 1))})
	                costs_test.append(testCost)
	            if  set_verbosity and epoch % 5 == 0:
	            	print('miniCost = '+str(minibatch_cost))
	            	testAccuracy = str(sess.run(accuracy, feed_dict={X: np.asarray(X_test), Y: np.asarray(Y_test).reshape((len(Y_test), 1))}))
	            	trainAccuracy = str(sess.run(accuracy, feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1))}))
	            	accuracy_train.append(float(trainAccuracy))
	            	accuracy_test.append(float(testAccuracy))
	            	print("Epoch "+str(epoch)+": \tTrain cost: "+str(trainCost)+" \tTest cost: "+str(testCost)+" \tTrain Accuracy: "+str(trainAccuracy)+" \tTest accuracy: "+str(testAccuracy))
	        

	        # Calculate accuracy on the test set
	        predict_op = tf.to_float(prediction > 0.5)
	        correct_prediction = tf.equal(predict_op, Y)
	        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	        train_accuracy = sess.run(accuracy, feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1))})
	        test_accuracy = sess.run(accuracy, feed_dict={X: np.asarray(X_test), Y: np.asarray(Y_test).reshape((len(Y_test), 1))})

	        if(set_verbosity):
	        	# Calculate the correct predictions
	        	plt.plot(np.arange(0, len(accuracy_train)*5, 5), accuracy_train,'b', label='accuracy_train')
	        	plt.plot(np.arange(0, len(accuracy_train)*5, 5), accuracy_test,'r', label='accuracy_test')
	        	plt.plot(range(0,len(costs_train)),costs_train,'--b', label='cost_train')
	        	plt.plot(range(0,len(costs_test)),costs_test,'--r', label='cost_test' )

	        	plt.ylabel('accuracy')
	        	plt.xlabel('epochs')
	        	plt.title("Learning rate =" + str(learning_rate))
	        	plt.legend()
	        	plt.show()


	        	print("Train Accuracy:", train_accuracy)
	        	print("Test Accuracy:", test_accuracy)
	        return (train_accuracy, test_accuracy)



	def modelSelection(folds, range_lstm_unit, range_news_hour):
	    import os
        # fold = list(train_i , dev_i)
        # train_i = (train_x, train_y) 
	    tf.logging.set_verbosity(tf.logging.ERROR)
	    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	    bestNumNews = 0
	    bestNumLSTMUnits = 0
	    bestAccuracy = 0
	    for numNews in range_news_hour:
		    for numLSTMUnits in range_lstm_unit:
			    print('---------- num_lstm_units: '+str(numLSTMUnits)+ ' news_per_hour: '+str(numNews))
			    k_fold_train_accuracy = []
			    k_fold_dev_accuracy = []

			    for fold in folds:
			        print("==", end="", flush=True)
			        (X_train, Y_train), (X_dev, Y_dev) = fold
			        (train_accuracy, dev_accuracy) = MyModel.model(X_train, Y_train, X_dev, Y_dev,  minibatch_size = batch_size,learning_rate = learning_rate, num_lstm_units = numLSTMUnits, news_per_hour = numNews, set_verbosity=False)
			        k_fold_train_accuracy.append(train_accuracy)
			        k_fold_dev_accuracy.append(dev_accuracy)
			    print("")
			    print('( train_fold_accuracy: '+str(sum(k_fold_train_accuracy) / len(folds))+ ', dev_fold_accuracy: '+str(sum(k_fold_dev_accuracy) / len(folds))+' )')
			    print('( train_fold_variance: '+str(np.var(np.asarray(k_fold_train_accuracy)))+ ', dev_fold_variance: '+str(np.var(np.asarray(k_fold_train_accuracy)))+' )')

			    if(dev_accuracy > bestAccuracy):
			        bestNumNews = numNews
			        bestNumLSTMUnits = numLSTMUnits

	    print('BEST: ( num_lstm_units: '+str(bestNumLSTMUnits)+ ', news_per_hour: '+str(numNews)+' )')










#  MAIN
# .
# .
# .
# .
# .
# .
# .
# .



Data.load_data(news_per_hour = 10, momentum_window=30, newsTimeToMarket = 20)

(X_train, Y_train), (X_test, Y_test) = Data.get_train_test_set()

test_x = tf.convert_to_tensor(np.asarray(X_test), dtype=tf.float32)
train_x = tf.convert_to_tensor(np.asarray(X_train), dtype=tf.float32)

train_y = tf.convert_to_tensor(np.asarray(Y_train), dtype=tf.float32)
test_y = tf.convert_to_tensor(np.asarray(Y_test), dtype=tf.float32)


print ("number of training examples = " + str(train_x.shape[0]))
print ("number of test examples = " + str(test_x.shape[0]))
print ("X_train shape: " + str(train_x.shape))
print ("Y_train shape: " + str(train_y.shape))
print ("X_test shape: " + str(test_x.shape))
print ("Y_test shape: " + str(test_y.shape))


print ("Test baseline: " + str( np.sum(np.asarray(Y_test)==1)/len(Y_test)))

MyModel.model(X_train, Y_train, X_test, Y_test,  minibatch_size = batch_size, learning_rate = learning_rate, num_lstm_units = num_lstm_units, news_per_hour = news_per_hour)


#folds = Data.get_cross_validation_train_dev_set(test_percentage=0.3, k_fold = 10,  dev_num_points=100)

#MyModel.modelSelection(folds, range_lstm_unit = np.arange(5, 250, 10), range_news_hour = np.arange(3, 15, 3))