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
num_lstm_units = 100





##
# 1 - non funziona con piu di un lstm layer
# 2 - controllare che lo stato sia propagato in maniera corretta
# 3 - controllare il modello in generale






num_lstm_layers = 1

#learning rate for adam
learning_rate=0.001


#size of batch 	
batch_size=128







class MyModel():

	## Managing state through batches:
	def get_state_variables(state_placeholder):
		l = tf.unstack(state_placeholder, axis=0)

		rnn_tuple_state = tuple(
		[tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
		 for idx in range(num_lstm_layers)]
		)



		return rnn_tuple_state

	def get_initial_state():
		if(num_lstm_layers == 1):
			return np.zeros([num_lstm_layers, 2, 1, num_lstm_units])

		return tuple([tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 1, num_lstm_units]), np.zeros([1, 1, num_lstm_units]))for idx in range(num_lstm_layers)])



	def create_placeholders(num_lstm_layers, num_lstm_units, news_per_hour):
	    X = tf.placeholder(tf.float32, shape=(None, news_per_hour, skip_vector_dim), name='X')
	    Y = tf.placeholder(tf.float32, shape=(None, n_y), name='Y')
	    lstm_state_placeholder = tf.placeholder(tf.float32, [num_lstm_layers, 2, None, num_lstm_units],  name='lstm_state')

	    return X, Y, lstm_state_placeholder



	def forward_propagation(X, init_state = None, num_lstm_units=num_lstm_units, num_lstm_layers=num_lstm_layers):

		if init_state != None:
			init_state = MyModel.get_state_variables(init_state)

	    # ATTENTION
		e = tf.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
		alpha = tf.nn.softmax(e, name='attention_weights')   												# tf.nn.softmax(logits,axis=None, ..)
		timeSlotEmbeddings =  tf.matmul(alpha, X, transpose_a=True, name='timeSlotEmbeddings')				# tf.matmul(a,b, transpose_a=False, transpose_b=False, name=None )

		# # LSTM
		# # (see https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)

		timeSlotSequence = timeSlotEmbeddings															# 1 sequenza di ? sample timeSlotEmbeddings.shape = (?, 1, 2400) -> memoria tra i vari sample
		#timeSlotSequence = tf.reshape(timeSlotEmbeddings, [100, -1, 2400])#In quale ordine li mette????															

		print(timeSlotSequence)

		# VEDERE COME FUNZIONA BACKPROP THROUGH TIME PER VEDERE QUALE E MEGLIO
		#timeSlotSequence = tf.transpose(timeSlotEmbeddings, perm=[1, 0, 2])								# ? sequenze di 1 sample -> NON c'e memoria esplicita tra i vari sample
	
		#lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm_units)											# Definisco il layer
		lstm_layer = tf.contrib.rnn.LSTMCell(num_lstm_units, use_peepholes=True)						# Definisco il layer

		lstm_network = tf.contrib.rnn.MultiRNNCell([lstm_layer] * num_lstm_layers)

		outputs, new_states = tf.nn.dynamic_rnn(lstm_network, timeSlotSequence,
			 initial_state=init_state,  dtype="float32",time_major=True)									# Definisco la rete ricorrente tramite il layer precedente

		#outputs = tf.reshape(outputs, [-1, num_lstm_units])  #In quale ordine li mette????
		outputs = tf.squeeze(outputs, axis=1)

		prediction = tf.layers.dense(outputs, 1, activation=tf.nn.sigmoid)


		return prediction, new_states



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
	          num_lstm_units = 100, num_lstm_layers=1, news_per_hour = 10, set_verbosity=True):
	    if(set_verbosity):
	    	print ("Learning rate: " +str(learning_rate))

	    ops.reset_default_graph()                      										# to be able to rerun the model without overwriting tf variables
	    m =int(len(X_train))
	    costs_train = []
	    costs_test = []
	    accuracy_train = []
	    accuracy_test = []

	    # Create Placeholders of the correct shape
	    X, Y, lstm_state_placeholder = MyModel.create_placeholders(num_lstm_layers=num_lstm_layers, num_lstm_units= num_lstm_units, news_per_hour=news_per_hour)

	    # Forward propagation: Build the forward propagation in the tensorflow graph
	    prediction, lstm_next_state = MyModel.forward_propagation(X, lstm_state_placeholder,num_lstm_layers=num_lstm_layers, num_lstm_units = num_lstm_units)	
	
	    
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
	        	# 1 perche per ora ho 1 sola sequenza
	            lstm_state = MyModel.get_initial_state()													#Ogni epoch reinizializzo stato

	            minibatch_cost = 0.0
	            num_minibatches = int(m / minibatch_size)
	            if(num_minibatches == 0):
	            	num_minibatches = 1
	            minibatches = MyModel.random_mini_batches(X_train, Y_train, minibatch_size)


	            for minibatch in minibatches:
	                # Select a minibatch
	                (minibatch_X, minibatch_Y) = minibatch
	                minibatch_X = np.asarray(minibatch_X)
	                minibatch_Y = np.asarray(minibatch_Y).reshape((len(minibatch_Y), 1))



	                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
	                _ , temp_cost, lstm_state = sess.run([optimizer, cost, lstm_next_state], feed_dict={X: minibatch_X, Y: minibatch_Y, lstm_state_placeholder: lstm_state})


	                # weights = tf.get_default_graph().get_tensor_by_name('dense_1/kernel:0')
	                #print('state: ' +str(new_state))

	                minibatch_cost += temp_cost / num_minibatches
	                
	            # Print the cost every epoch
	            if  epoch % 1 == 0:

	                lstm_temp_state = MyModel.get_initial_state()
	                trainCost, lstm_temp_state = sess.run([cost, lstm_next_state], feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1)), lstm_state_placeholder: lstm_temp_state})
	                costs_train.append(trainCost)
	                testCost = sess.run(cost, feed_dict={X: np.asarray(X_test), Y: np.asarray(Y_test).reshape((len(Y_test), 1)), lstm_state_placeholder: lstm_temp_state})
	                costs_test.append(testCost)

	                lstm_temp_state  = MyModel.get_initial_state()
	                trainAccuracy, lstm_temp_state = sess.run([accuracy, lstm_next_state], feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1)),lstm_state_placeholder: lstm_temp_state})
	                testAccuracy = sess.run(accuracy, feed_dict={X: np.asarray(X_test), Y: np.asarray(Y_test).reshape((len(Y_test), 1)),lstm_state_placeholder: lstm_temp_state})
	                accuracy_train.append(float(trainAccuracy))
	                accuracy_test.append(float(testAccuracy))
	            if  set_verbosity and epoch % 5 == 0:
	            	print('miniCost ='+str(minibatch_cost))

	            	print("Epoch "+str(epoch)+": \tTrain cost: "+str(trainCost)+" \tTest cost: "+str(testCost)+" \tTrain Accuracy: "+str(trainAccuracy)+" \tTest accuracy: "+str(testAccuracy))
        



	        lstm_temp_state  = MyModel.get_initial_state()
	        trainAccuracy, lstm_temp_state = sess.run([accuracy, lstm_next_state], feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1)),lstm_state_placeholder: lstm_temp_state})
	        testAccuracy = sess.run(accuracy, feed_dict={X: np.asarray(X_test), Y: np.asarray(Y_test).reshape((len(Y_test), 1)),lstm_state_placeholder: lstm_temp_state})
	        
	        if(set_verbosity):
	        	# plt.plot(np.arange(0, len(accuracy_train)*5, 5), accuracy_train,'b', label='accuracy_train')
	        	# plt.plot(np.arange(0, len(accuracy_train)*5, 5), accuracy_test,'r', label='accuracy_test')
	        	plt.plot(range(0,len(accuracy_train)), accuracy_train,'b', label='accuracy_train')
	        	plt.plot(range(0,len(accuracy_train)), accuracy_test,'r', label='accuracy_test')
	        	plt.plot(range(0,len(costs_train)),costs_train,'--b', label='cost_train')
	        	plt.plot(range(0,len(costs_test)),costs_test,'--r', label='cost_test' )

	        	plt.ylabel('accuracy')
	        	plt.xlabel('epochs')
	        	plt.title("Learning rate =" + str(learning_rate))
	        	plt.legend()
	        	plt.show()

	        	print("Train Accuracy:", max(accuracy_train))
	        	print("Test Accuracy:", max(accuracy_test))
                
	        return (trainAccuracy, testAccuracy)



	def modelSelection(folds, range_lstm_unit, range_news_hour):
		import os
		# fold = list(train_i , dev_i)
		# train_i = (train_x, train_y) 
		tf.logging.set_verbosity(tf.logging.ERROR)
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

		bestNumLSTMUnits = 0
		bestAccuracy = 0
		for numLSTMUnits in range_lstm_unit:
		    print('---------- num_lstm_units: '+str(numLSTMUnits))
		    k_fold_train_accuracy = []
		    k_fold_dev_accuracy = []
		    always_yes = []

		    for fold in folds:
		        print("==", end="", flush=True)
		        (X_train, Y_train), (X_dev, Y_dev) = fold
		        (train_accuracy, dev_accuracy) = MyModel.model(X_train, Y_train, X_dev, Y_dev,  minibatch_size = batch_size,learning_rate = learning_rate, num_lstm_units = numLSTMUnits, set_verbosity=False)
		        k_fold_train_accuracy.append(train_accuracy)
		        k_fold_dev_accuracy.append(dev_accuracy)
		        always_yes.append(np.sum(np.asarray(Y_dev)==1)/len(Y_dev))
		    print('( train_fold_accuracy: '+str(sum(k_fold_train_accuracy) / len(folds))+ ', dev_fold_accuracy: '+str(sum(k_fold_dev_accuracy) / len(folds))+' )')
		    print('Dev accuracies: '+str(k_fold_dev_accuracy))
		    print('Dev predict y=1: '+str(always_yes))
		    print('( train_fold_variance: '+str(np.var(np.asarray(k_fold_train_accuracy)))+ ', dev_fold_variance: '+str(np.var(np.asarray(k_fold_dev_accuracy)))+' )')

		    if(dev_accuracy > bestAccuracy):
		        bestNumLSTMUnits = numLSTMUnits

		print('BEST: ( num_lstm_units: '+str(bestNumLSTMUnits)+ ')')










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

print('.........................')
print ("number of training examples = " + str(train_x.shape[0]))
print ("number of test examples = " + str(test_x.shape[0]))
print ("X_train shape: " + str(train_x.shape))
print ("Y_train shape: " + str(train_y.shape))
print ("X_test shape: " + str(test_x.shape))
print ("Y_test shape: " + str(test_y.shape))
print('.........................')



print ("Test baseline: " + str( np.sum(np.asarray(Y_test)==1)/len(Y_test)))

MyModel.model(X_train, Y_train, X_test, Y_test,  minibatch_size = batch_size, learning_rate = learning_rate,num_lstm_layers=num_lstm_layers, num_lstm_units = num_lstm_units, news_per_hour = news_per_hour)


# folds = Data.get_cross_validation_train_dev_set(test_percentage=0.3, k_fold = 10,  dev_num_points=100)
# MyModel.modelSelection(folds, range_lstm_unit = np.arange(5, 250, 10), range_news_hour = np.arange(3, 15, 3))







'''
.........................
number of training examples = 1150
number of test examples = 493
X_train shape: (1150, 10, 2400)
Y_train shape: (1150,)
X_test shape: (493, 10, 2400)
Y_test shape: (493,)
.........................
Test baseline: 0.5841784989858012
Fold lenght: 105
---------- num_lstm_units: 5
====================( train_fold_accuracy: 0.8685714364051819, dev_fold_accuracy: 0.5269999966025353 )
Dev accuracies: [1.0, 0.64, 0.8, 0.18, 0.7, 0.33, 0.48, 0.77, 0.26, 0.11]
( train_fold_variance: 0.01106213, dev_fold_variance: 0.080661 )
---------- num_lstm_units: 15
====================( train_fold_accuracy: 0.8942857205867767, dev_fold_accuracy: 0.539999996125698 )
Dev accuracies: [1.0, 0.64, 0.8, 0.18, 0.7, 0.42, 0.48, 0.81, 0.26, 0.11]
( train_fold_variance: 0.012398185, dev_fold_variance: 0.079859994 )
---------- num_lstm_units: 25
====================( train_fold_accuracy: 0.8895238161087036, dev_fold_accuracy: 0.5339999988675117 )
Dev accuracies: [1.0, 0.64, 0.8, 0.18, 0.7, 0.44, 0.48, 0.73, 0.26, 0.11]
( train_fold_variance: 0.011976415, dev_fold_variance: 0.07570399 )
---------- num_lstm_units: 35
====================( train_fold_accuracy: 0.8952381014823914, dev_fold_accuracy: 0.5200000002980232 )
Dev accuracies: [1.0, 0.64, 0.79, 0.18, 0.7, 0.54, 0.48, 0.5, 0.26, 0.11]
( train_fold_variance: 0.012480724, dev_fold_variance: 0.070379995 )
---------- num_lstm_units: 45
====================( train_fold_accuracy: 0.9019047677516937, dev_fold_accuracy: 0.5399999991059303 )
Dev accuracies: [1.0, 0.64, 0.79, 0.18, 0.7, 0.5, 0.48, 0.74, 0.26, 0.11]
( train_fold_variance: 0.0131709725, dev_fold_variance: 0.07478 )
---------- num_lstm_units: 55
====================( train_fold_accuracy: 0.9057142913341523, dev_fold_accuracy: 0.5539999976754189 )
Dev accuracies: [1.0, 0.64, 0.79, 0.18, 0.7, 0.67, 0.48, 0.71, 0.26, 0.11]
( train_fold_variance: 0.013268933, dev_fold_variance: 0.075004 )
---------- num_lstm_units: 65
====================( train_fold_accuracy: 0.8990476250648498, dev_fold_accuracy: 0.540999998152256 )
Dev accuracies: [1.0, 0.64, 0.79, 0.18, 0.7, 0.55, 0.48, 0.7, 0.26, 0.11]
( train_fold_variance: 0.012792742, dev_fold_variance: 0.073189 )
---------- num_lstm_units: 75
====================( train_fold_accuracy: 0.9057142913341523, dev_fold_accuracy: 0.5120000019669533 )
Dev accuracies: [0.99, 0.64, 0.79, 0.18, 0.7, 0.54, 0.48, 0.43, 0.26, 0.11]
( train_fold_variance: 0.0135773225, dev_fold_variance: 0.070135996 )
---------- num_lstm_units: 85
====================( train_fold_accuracy: 0.9066666722297668, dev_fold_accuracy: 0.5479999974370002 )
Dev accuracies: [0.99, 0.64, 0.79, 0.18, 0.7, 0.64, 0.48, 0.69, 0.26, 0.11]
( train_fold_variance: 0.013438547, dev_fold_variance: 0.072896 )
---------- num_lstm_units: 95
====================( train_fold_accuracy: 0.9076190531253815, dev_fold_accuracy: 0.5279999986290932 )
Dev accuracies: [0.97, 0.64, 0.79, 0.18, 0.7, 0.56, 0.48, 0.59, 0.26, 0.11]
( train_fold_variance: 0.013443081, dev_fold_variance: 0.068096 )
---------- num_lstm_units: 105
====================( train_fold_accuracy: 0.908571434020996, dev_fold_accuracy: 0.49999999552965163 )
Dev accuracies: [0.99, 0.64, 0.65, 0.18, 0.7, 0.49, 0.48, 0.5, 0.26, 0.11]
( train_fold_variance: 0.012992288, dev_fold_variance: 0.063480005 )
---------- num_lstm_units: 115
====================( train_fold_accuracy: 0.9228571474552154, dev_fold_accuracy: 0.509999991953373 )
Dev accuracies: [0.95, 0.64, 0.78, 0.18, 0.7, 0.69, 0.48, 0.31, 0.26, 0.11]
( train_fold_variance: 0.012470746, dev_fold_variance: 0.07241999 )
---------- num_lstm_units: 125
====================( train_fold_accuracy: 0.9114285767078399, dev_fold_accuracy: 0.5279999986290932 )
Dev accuracies: [0.98, 0.64, 0.79, 0.18, 0.7, 0.7, 0.48, 0.44, 0.26, 0.11]
( train_fold_variance: 0.01367891, dev_fold_variance: 0.072235994 )
---------- num_lstm_units: 135
====================( train_fold_accuracy: 0.9095238149166107, dev_fold_accuracy: 0.4979999974370003 )
Dev accuracies: [0.95, 0.64, 0.8, 0.18, 0.7, 0.55, 0.48, 0.31, 0.26, 0.11]
( train_fold_variance: 0.013138321, dev_fold_variance: 0.070316 )
---------- num_lstm_units: 145
====================( train_fold_accuracy: 0.9095238149166107, dev_fold_accuracy: 0.5069999977946281 )
Dev accuracies: [0.99, 0.64, 0.8, 0.18, 0.7, 0.46, 0.48, 0.45, 0.26, 0.11]
( train_fold_variance: 0.013374148, dev_fold_variance: 0.070581 )
---------- num_lstm_units: 155
====================( train_fold_accuracy: 0.919047623872757, dev_fold_accuracy: 0.5259999975562095 )
Dev accuracies: [0.99, 0.64, 0.79, 0.18, 0.7, 0.64, 0.48, 0.47, 0.26, 0.11]
( train_fold_variance: 0.012448977, dev_fold_variance: 0.071003996 )
---------- num_lstm_units: 165
====================( train_fold_accuracy: 0.9104761958122254, dev_fold_accuracy: 0.5100000008940697 )
Dev accuracies: [0.98, 0.64, 0.79, 0.18, 0.7, 0.63, 0.48, 0.33, 0.26, 0.11]
( train_fold_variance: 0.013282537, dev_fold_variance: 0.07314 )
---------- num_lstm_units: 175
====================( train_fold_accuracy: 0.9228571474552154, dev_fold_accuracy: 0.5260000005364418 )
Dev accuracies: [0.99, 0.64, 0.8, 0.18, 0.7, 0.6, 0.48, 0.5, 0.26, 0.11]
( train_fold_variance: 0.012434466, dev_fold_variance: 0.070544 )
---------- num_lstm_units: 185
====================( train_fold_accuracy: 0.9228571474552154, dev_fold_accuracy: 0.5109999999403954 )
Dev accuracies: [0.99, 0.64, 0.79, 0.18, 0.7, 0.62, 0.48, 0.34, 0.26, 0.11]
( train_fold_variance: 0.012470746, dev_fold_variance: 0.073509 )
---------- num_lstm_units: 195
====================( train_fold_accuracy: 0.9238095283508301, dev_fold_accuracy: 0.5009999975562096 )
Dev accuracies: [1.0, 0.64, 0.79, 0.18, 0.7, 0.51, 0.48, 0.34, 0.26, 0.11]
( train_fold_variance: 0.012462584, dev_fold_variance: 0.073189 )
---------- num_lstm_units: 205
====================( train_fold_accuracy: 0.9257142901420593, dev_fold_accuracy: 0.506999994814396 )
Dev accuracies: [0.95, 0.64, 0.78, 0.18, 0.7, 0.67, 0.48, 0.3, 0.26, 0.11]
( train_fold_variance: 0.012731065, dev_fold_variance: 0.07214099 )
---------- num_lstm_units: 215
====================( train_fold_accuracy: 0.926666671037674, dev_fold_accuracy: 0.5219999983906746 )
Dev accuracies: [0.99, 0.64, 0.8, 0.18, 0.7, 0.62, 0.48, 0.44, 0.26, 0.11]
( train_fold_variance: 0.012826303, dev_fold_variance: 0.071536 )
---------- num_lstm_units: 225
====================( train_fold_accuracy: 0.9247619092464447, dev_fold_accuracy: 0.5429999962449074 )
Dev accuracies: [1.0, 0.64, 0.8, 0.19, 0.7, 0.69, 0.48, 0.56, 0.26, 0.11]
( train_fold_variance: 0.012724715, dev_fold_variance: 0.07270099 )
---------- num_lstm_units: 235
====================( train_fold_accuracy: 0.9228571474552154, dev_fold_accuracy: 0.5239999994635582 )
Dev accuracies: [0.97, 0.64, 0.79, 0.22, 0.7, 0.58, 0.48, 0.49, 0.26, 0.11]
( train_fold_variance: 0.012543309, dev_fold_variance: 0.06538401 )
---------- num_lstm_units: 245
====================( train_fold_accuracy: 0.9200000047683716, dev_fold_accuracy: 0.5089999929070472 )
Dev accuracies: [0.98, 0.64, 0.78, 0.18, 0.7, 0.52, 0.48, 0.44, 0.26, 0.11]
( train_fold_variance: 0.013137413, dev_fold_variance: 0.068409 )
BEST: ( num_lstm_units: 245)

'''

