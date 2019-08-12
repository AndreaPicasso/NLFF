import pandas as pd
import numpy as np
import tensorflow as tf
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


#hidden LSTM units
num_lstm_units=100

#learning rate for adam
learning_rate=0.001


#size of batch 	
batch_size=128
batch_size=2000







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

		# # LSTM
		# # (see https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)

		timeSlotSequence = timeSlotEmbeddings															# 1 sequenza di ? sample timeSlotEmbeddings.shape = (?, 1, 2400) -> memoria tra i vari sample

		# timeSlotSequence.append(timeSlotEmbeddings)													# ? sequenze di 1 sample -> NON CREDO ci sia memoria tra i vari sample
	
		lstm_layer= tf.contrib.rnn.BasicLSTMCell(num_lstm_units)										# Definisco il layer
		outputs,_= tf.nn.dynamic_rnn(lstm_layer, timeSlotSequence, dtype="float32",time_major=True)		# Definisco la rete ricorrente tramite il layer precedente

		outputs = tf.squeeze(outputs, axis=1)	
		prediction = tf.layers.dense(outputs, 1, activation=tf.nn.sigmoid)

		return prediction



	def compute_cost(Y_hat, Y):
	    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y))		# Y * -log(sigmoid(Y_hat)) + (1 - Y) * -log(1 - sigmoid(Y_hat))
	    #cost = tf.reduce_mean(tf.squared_difference(Y_hat, Y))
	    return cost



	def random_mini_batches(X_train, Y_train, minibatch_size):
		minibatches = list()

		m = int(len(X_train))
		if(minibatch_size > m):
			minibatches.append((X_train, Y_train))
			return minibatches

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
	            	print('miniCost ='+str(minibatch_cost))
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
	        	plt.plot(np.arange(0, len(accuracy_train)*5, 5), accuracy_train,'b')
	        	plt.plot(np.arange(0, len(accuracy_train)*5, 5), accuracy_test,'r')
	        	plt.ylabel('accuracy')
	        	plt.xlabel('epochs')
	        	plt.title("Learning rate =" + str(learning_rate))
	        	plt.show()
	        	# plot the cost
	        	plt.plot(range(0,len(costs_train)),costs_train,'b',costs_test,'r')
	        	plt.ylabel('cost')
	        	plt.xlabel('epochs')
	        	plt.title("Learning rate =" + str(learning_rate))
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
			    print('( train_fold_accuracy: '+str(sum(k_fold_train_accuracy) / len(folds))+ ', dev_fold_accuracy: '+str(sum(k_fold_dev_accuracy) / len(folds))+' )')
			    print('Dev accuracies: '+str(k_fold_dev_accuracy))
			    print('( train_fold_variance: '+str(np.var(np.asarray(k_fold_train_accuracy)))+ ', dev_fold_variance: '+str(np.var(np.asarray(k_fold_dev_accuracy)))+' )')

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

#MyModel.model(X_train, Y_train, X_test, Y_test,  minibatch_size = batch_size, learning_rate = learning_rate, num_lstm_units = num_lstm_units, news_per_hour = news_per_hour)


folds = Data.get_cross_validation_train_dev_set(test_percentage=0.3, k_fold = 10,  dev_num_points=100)

MyModel.modelSelection(folds, range_lstm_unit = np.arange(5, 250, 10), range_news_hour = np.arange(3, 15, 3))
















'''
MODEL SELECTION RESULTS:


---------- num_lstm_units: 5 news_per_hour: 3
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5079999908804893 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 15 news_per_hour: 3
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.5089999929070472 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 25 news_per_hour: 3
====================
( train_fold_accuracy: 0.9057142913341523, dev_fold_accuracy: 0.5089999958872795 )
( train_fold_variance: 0.013232651, dev_fold_variance: 0.013232651 )
---------- num_lstm_units: 35 news_per_hour: 3
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5039999976754188 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 45 news_per_hour: 3
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.5169999971985817 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288 )
---------- num_lstm_units: 55 news_per_hour: 3
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.4989999994635582 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288 )
---------- num_lstm_units: 65 news_per_hour: 3
====================
( train_fold_accuracy: 0.9095238149166107, dev_fold_accuracy: 0.49499999135732653 )
( train_fold_variance: 0.012829931, dev_fold_variance: 0.012829931 )
---------- num_lstm_units: 75 news_per_hour: 3
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.48199999183416364 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 85 news_per_hour: 3
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.49399999529123306 )
( train_fold_variance: 0.013282537, dev_fold_variance: 0.013282537 )
---------- num_lstm_units: 95 news_per_hour: 3
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.5019999995827675 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288 )
---------- num_lstm_units: 105 news_per_hour: 3
====================
( train_fold_accuracy: 0.9057142913341523, dev_fold_accuracy: 0.507999999821186 )
( train_fold_variance: 0.013033104, dev_fold_variance: 0.013033104 )
---------- num_lstm_units: 115 news_per_hour: 3
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5109999969601631 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 125 news_per_hour: 3
====================
( train_fold_accuracy: 0.9095238149166107, dev_fold_accuracy: 0.4849999979138374 )
( train_fold_variance: 0.012829931, dev_fold_variance: 0.012829931 )
---------- num_lstm_units: 135 news_per_hour: 3
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.5039999961853028 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 145 news_per_hour: 3
====================
( train_fold_accuracy: 0.9095238149166107, dev_fold_accuracy: 0.49799999594688416 )
( train_fold_variance: 0.01310204, dev_fold_variance: 0.01310204 )
---------- num_lstm_units: 155 news_per_hour: 3
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.4999999925494194 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 165 news_per_hour: 3
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.49999999552965163 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 175 news_per_hour: 3
====================
( train_fold_accuracy: 0.9095238149166107, dev_fold_accuracy: 0.5129999980330467 )
( train_fold_variance: 0.013482993, dev_fold_variance: 0.013482993 )
---------- num_lstm_units: 185 news_per_hour: 3
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.46499999463558195 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 195 news_per_hour: 3
====================
( train_fold_accuracy: 0.9057142913341523, dev_fold_accuracy: 0.48999999910593034 )
( train_fold_variance: 0.013341496, dev_fold_variance: 0.013341496 )
---------- num_lstm_units: 205 news_per_hour: 3
====================
( train_fold_accuracy: 0.9104761958122254, dev_fold_accuracy: 0.5109999939799309 )
( train_fold_variance: 0.013300678, dev_fold_variance: 0.013300678 )
---------- num_lstm_units: 215 news_per_hour: 3
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.4589999929070473 )
( train_fold_variance: 0.013318819, dev_fold_variance: 0.013318819 )
---------- num_lstm_units: 225 news_per_hour: 3
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.4899999961256981 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288 )
---------- num_lstm_units: 235 news_per_hour: 3
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.4769999966025352 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 245 news_per_hour: 3
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5009999945759773 )
( train_fold_variance: 0.013515646, dev_fold_variance: 0.013515646 )
---------- num_lstm_units: 5 news_per_hour: 6
====================
( train_fold_accuracy: 0.9095238149166107, dev_fold_accuracy: 0.5070000007748604 )
( train_fold_variance: 0.013482993, dev_fold_variance: 0.013482993 )
---------- num_lstm_units: 15 news_per_hour: 6
====================
( train_fold_accuracy: 0.9057142913341523, dev_fold_accuracy: 0.4959999963641167 )
( train_fold_variance: 0.0135773225, dev_fold_variance: 0.0135773225 )
---------- num_lstm_units: 25 news_per_hour: 6
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.4689999967813492 )
( train_fold_variance: 0.013282537, dev_fold_variance: 0.013282537 )
---------- num_lstm_units: 35 news_per_hour: 6
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.5070000007748604 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288 )
---------- num_lstm_units: 45 news_per_hour: 6
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.5059999987483025 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 55 news_per_hour: 6
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.49499999433755876 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 65 news_per_hour: 6
====================
( train_fold_accuracy: 0.9104761958122254, dev_fold_accuracy: 0.5299999967217446 )
( train_fold_variance: 0.012919727, dev_fold_variance: 0.012919727 )
---------- num_lstm_units: 75 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.4839999958872795 )
( train_fold_variance: 0.013896597, dev_fold_variance: 0.013896597 )
---------- num_lstm_units: 85 news_per_hour: 6
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.5119999945163727 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 95 news_per_hour: 6
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.5049999997019767 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288 )
---------- num_lstm_units: 105 news_per_hour: 6
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.5 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 115 news_per_hour: 6
====================
( train_fold_accuracy: 0.9161904811859131, dev_fold_accuracy: 0.5379999950528145 )
( train_fold_variance: 0.012078002, dev_fold_variance: 0.012078002 )
---------- num_lstm_units: 125 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5040000006556511 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 135 news_per_hour: 6
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.5029999986290932 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 145 news_per_hour: 6
====================
( train_fold_accuracy: 0.9114285767078399, dev_fold_accuracy: 0.486999998986721 )
( train_fold_variance: 0.013134693, dev_fold_variance: 0.013134693 )
---------- num_lstm_units: 155 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.4769999966025352 )
( train_fold_variance: 0.013152833, dev_fold_variance: 0.013152833 )
---------- num_lstm_units: 165 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.4980000004172325 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 175 news_per_hour: 6
====================
( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.49999999850988386 )
( train_fold_variance: 0.013365986, dev_fold_variance: 0.013365986 )
---------- num_lstm_units: 185 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5029999986290932 )
( train_fold_variance: 0.013896597, dev_fold_variance: 0.013896597 )
---------- num_lstm_units: 195 news_per_hour: 6
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.507999999821186 )
( train_fold_variance: 0.013119273, dev_fold_variance: 0.013119273 )
---------- num_lstm_units: 205 news_per_hour: 6
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.5019999980926514 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288 )
---------- num_lstm_units: 215 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.49399999529123306 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 225 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5129999950528145 )
( train_fold_variance: 0.013515646, dev_fold_variance: 0.013515646 )
---------- num_lstm_units: 235 news_per_hour: 6
====================
( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.4769999966025352 )
( train_fold_variance: 0.013170974, dev_fold_variance: 0.013170974 )
---------- num_lstm_units: 245 news_per_hour: 6
====================
( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.48399999290704726 )
( train_fold_variance: 0.012992288, dev_fold_variance: 0.012992288


'''