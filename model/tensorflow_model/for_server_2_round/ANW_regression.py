
import pandas as pd
import numpy as np
import tensorflow as tf
import math
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors



# Attention not only on one news but news are aggregated
# as sum of words in one hour, a window of different hours is taken
# To "balance" the classes the cost is computed weighting more the less common class
# 1 is the cost of the neg class, the cost of the pos class is computed as = #neg/#pos
# .
# .
# . Experiment with quantile, see which is the accuracy for each quantile of change
# . Two settings:
# . - cost considered as MSE() on normalized % of change ->  regression WHEIGHT SUCH THAT PREDICT GOOD HIGH FLUCTUATIONS IS MORE IMPORTANT
# . - cost considered as cross_entropy() ->  classif
# .
# .






skip_vector_dim = 8
n_y = 1 #Numero di output, Per ora sali / scendi poi metteremo neutrale



def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        #return -1
        return 0
    
class Data():
    X = []
    Y = []
    #positive_unbalancing_rate = 1

    def get_train_test_set(test_percentage=0.3):
        idx_split = math.floor(len(Data.X)*(1-test_percentage))

        train_x = Data.X[:idx_split]
        train_y = Data.Y[:idx_split]
        test_x = Data.X[idx_split:]
        test_y = Data.Y[idx_split:]

        return (train_x, train_y), (test_x, test_y)



    def get_cross_validation_train_dev_set(test_percentage=0.3, k_fold = 10,  dev_num_points=100):

        # https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection/14109#14109


        (train_x, train_y), _  = Data.get_train_test_set(test_percentage=test_percentage)
        m = int(len(train_x))
        samples_per_fold = int( (m - dev_num_points) / k_fold)
        # print('Fold lenght: '+str(samples_per_fold))
        fold = list()
        index = 0
        while(len(fold) < k_fold):
            fold.append(((train_x[index:index+samples_per_fold], train_y[index:index+samples_per_fold]),
                    (train_x[index+samples_per_fold:index+samples_per_fold+dev_num_points], train_y[index+samples_per_fold:index+samples_per_fold+dev_num_points])))
            index += samples_per_fold

        return fold





    def load_data(ticker='AAPL', momentum_window=28, newsTimeToMarket = 0, X_window_average=None, set_verbosity=True):
        X_path = '/home/simone/Desktop/NLFF/intrinioDatasetUpdated/SentimentFullAggregatedHourly/'+str(ticker)+'.csv'
        Y_path = '/home/simone/Desktop/NLFF/indexes/indexes'+str(ticker)+'.csv'
        
        
        
        if(set_verbosity):
            print('Reading dataset...')
            
        x = pd.read_csv(X_path)
        x = x.rename(index=str, columns={"initTime": "PUBLICATION_DATE"})
        #cambio l'ordine dalla piu vecchia alla piu recente
        if(set_verbosity):
            print('Ordering dataset...')
        x = x.sort_values(by=['PUBLICATION_DATE'])
        x = x.reset_index(drop=True)
        
        if(X_window_average != None):
            if(set_verbosity):
                print('Moving average..')
            x['CONSTRAINING'] = x['CONSTRAINING'].rolling(window=X_window_average,center=False).mean()
            x['LITIGIOUS'] = x['LITIGIOUS'].rolling(window=X_window_average,center=False).mean()
            x['NEGATIVE'] = x['NEGATIVE'].rolling(window=X_window_average,center=False).mean()
            x['POSITIVE'] = x['POSITIVE'].rolling(window=X_window_average,center=False).mean()
            x['UNCERTAINTY'] = x['UNCERTAINTY'].rolling(window=X_window_average,center=False).mean()
            x['SUPERFLUOUS'] = x['SUPERFLUOUS'].rolling(window=X_window_average,center=False).mean()
            x['INTERESTING'] = x['INTERESTING'].rolling(window=X_window_average,center=False).mean()
            x['NUM_NEWS'] = x['NUM_NEWS'].rolling(window=X_window_average,center=False).mean()

            x.drop(np.arange(X_window_average-1), inplace=True)
            x = x.reset_index(drop=True)


        # #Normalizzo
        # min_max_scaler = preprocessing.MinMaxScaler()
        # x[['CONSTRAINING', 'LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY', 'SUPERFLUOUS','INTERESTING']] = min_max_scaler.fit_transform(x[['CONSTRAINING', 'LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY','SUPERFLUOUS','INTERESTING']].values)
        
            

        for i, row in x.iterrows():
            x.at[i,'PUBLICATION_DATE'] =datetime.strptime(x['PUBLICATION_DATE'][i], '%Y-%m-%d %H:%M:%S')

            
            
            
        y = pd.read_csv(Y_path)
        y = y.rename(index=str, columns={"date": "DATE"})

        #PER ORA SCARTO GLI INDICI, POI SARA' DA METTERLI DENTRO X
        #y = y['DATE', 'close']
        for i, row in y.iterrows():
            y['DATE'].at[i] = datetime.strptime(y['DATE'][i], '%Y-%m-%d %H:%M:%S') 

        z = list()
        if(set_verbosity):
            print('y(t) - y(t-1) ...')
        #PAST
        for i in range(momentum_window-newsTimeToMarket, y.shape[0]-newsTimeToMarket):
            z.append((y['close'][i+newsTimeToMarket] - y['close'][i-momentum_window+newsTimeToMarket])/y['close'][i-momentum_window+newsTimeToMarket])

        y = y.reset_index(drop=True)
        y.drop(np.arange(0, momentum_window), inplace=True)
        y = y.reset_index(drop=True)
        y['labels'] = [sign(entry) for entry in z]
        y['perc'] = z
        y = y.reset_index(drop=True)


        
        if(set_verbosity):
            print('Alligning dataset and constructing cube..')

        initDate = max(y['DATE'][0], x['PUBLICATION_DATE'][0])
        finalDate = min(y['DATE'][len(y)-1], x['PUBLICATION_DATE'][len(x)-1])

        y.drop(y[y.DATE > finalDate].index, inplace=True)
        y.drop(y[y.DATE < initDate].index, inplace=True)
        y = y.reset_index(drop=True)
        x.drop(x[x.PUBLICATION_DATE > finalDate].index, inplace=True)
        x.drop(x[x.PUBLICATION_DATE < initDate].index, inplace=True)
        x = x.reset_index(drop=True)

        # if len(x) > len(y):
        #     print([date if date not in y['DATE'] else '' for date in x['PUBLICATION_DATE']])

        # if len(x) < len(y):
        #     print([date if date not in x['PUBLICATION_DATE']  else '' for date in y['DATE']])


        #Normalizzo
        min_max_scaler = preprocessing.MinMaxScaler()
        Xtemp = min_max_scaler.fit_transform(x[['CONSTRAINING', 'LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY','SUPERFLUOUS','INTERESTING','NUM_NEWS']].values)
        
        m = min(y['perc'])
        M = max(y['perc'])
        y['perc'] = 2*(y['perc'] - m)/(M-m)-1

        if(set_verbosity):
            plt.hist(y['perc'].tolist())
            plt.title('Fluctiations percentage distribution ')
            plt.show()



        #Create window of 30 hours
        X = list()
        for i in range(30,len(Xtemp)):
            X.append(Xtemp[i-30:i])

        Y = y['labels'][30:].tolist()
        Z = y['perc'][30:].tolist()

        # y_volatility = np.absolute(Z)
        # quantile_boundaries = np.linspace(min(y_volatility), max(y_volatility), num=4)
        # quantile = list()
        # for i in range(len(y_volatility)):
        #     if(y_volatility[i] < quantile_boundaries[1]):
        #         quantile.append(1)
        #     elif (y_volatility[i] >= quantile_boundaries[1] and y_volatility[i] < quantile_boundaries[2]):
        #         quantile.append(2)
        #     elif (y_volatility[i] >= quantile_boundaries[2] and y_volatility[i] < quantile_boundaries[3]):
        #         quantile.append(3)
        #     else:
        #         quantile.append(4)


        assert len(X) == len(Y) == len(Z)

        Data.X = np.asarray(X)
        Data.Y = np.asarray(Z)




class MyModel():
    
    def __init__(self, num_lstm_units, num_lstm_layers, news_per_hour, learning_rate = 0.009,
              num_epochs = 30, minibatch_size = 128):
        self.num_lstm_units = num_lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.news_per_hour = news_per_hour
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        
        

    ## Managing state through batches:
    def get_state_variables(self, state_placeholder):
        l = tf.unstack(state_placeholder, axis=0)
        rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(l[idx][0], l[idx][1])
         for idx in range(self.num_lstm_layers)]
        )
        return rnn_tuple_state

    def get_initial_state(self):
        if(self.num_lstm_layers == 1):
            return np.zeros([self.num_lstm_layers, 2, 1, self.num_lstm_units])

        return tuple([tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 1, self.num_lstm_units]), np.zeros([1, 1, self.num_lstm_units]))for idx in range(self.num_lstm_layers)])



    def create_placeholders(self):
        X = tf.placeholder(tf.float32, shape=(None, self.news_per_hour, skip_vector_dim), name='X')
        Y = tf.placeholder(tf.float32, shape=(None, n_y), name='Y')
        lstm_state_placeholder = tf.placeholder(tf.float32, [self.num_lstm_layers, 2, None, self.num_lstm_units],  name='lstm_state')
        
        return X, Y, lstm_state_placeholder



    def forward_propagation(self, X, init_state = None):

        if init_state != None:
            init_state = self.get_state_variables(init_state)

        # ATTENTION
        e = tf.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
        alpha = tf.nn.softmax(e, name='attention_weights')                                                  # tf.nn.softmax(logits,axis=None, ..)
        timeSlotEmbeddings =  tf.matmul(alpha, X, transpose_a=True, name='timeSlotEmbeddings')              # tf.matmul(a,b, transpose_a=False, transpose_b=False, name=None )

        # # LSTM
        # # (see https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)

        timeSlotSequence = timeSlotEmbeddings                                                           # 1 sequenza di ? sample timeSlotEmbeddings.shape = (?, 1, 2400) -> memoria tra i vari sample


        # VEDERE COME FUNZIONA BACKPROP THROUGH TIME PER VEDERE QUALE E MEGLIO
        #timeSlotSequence = tf.transpose(timeSlotEmbeddings, perm=[1, 0, 2])                                # ? sequenze di 1 sample -> NON c'e memoria esplicita tra i vari sample

        #lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm_units)                                          # Definisco il layer
        lstm_layer = tf.contrib.rnn.LSTMCell(self.num_lstm_units, use_peepholes=True)                       # Definisco il layer

        lstm_network = tf.contrib.rnn.MultiRNNCell([lstm_layer] * self.num_lstm_layers)


        outputs, new_states = tf.nn.dynamic_rnn(lstm_network, timeSlotSequence,
             initial_state=init_state,  dtype="float32",time_major=True)                                    # Definisco la rete ricorrente tramite il layer precedente


        outputs = tf.squeeze(outputs, axis=1)
    #       outputs = tf.squeeze(outputs, axis=0)

        prediction = tf.layers.dense(outputs, 1, activation=tf.nn.tanh)


        return prediction, new_states



    def compute_cost(self, Y_hat, Y):        
        # abs(Y) * (Y - Y_hat).*(Y - Y_hat)       abs(Y) works as weights, making errors on bigger fluctuations is weighted more
        w = tf.clip_by_value(tf.abs(Y), clip_value_min=0.15, clip_value_max=1)
        cost = tf.losses.mean_squared_error(Y, Y_hat, weights = w)
        # cost = tf.losses.mean_squared_error(Y, Y_hat)

        return cost



    def mini_batches(self, X_train, Y_train):
        minibatches = list()

        m = int(len(X_train))
        if(self.minibatch_size > m):
            minibatches.append((X_train, Y_train))
            return minibatches

        minibatches.append((X_train[0:self.minibatch_size], Y_train[0:self.minibatch_size]))
        iterSize = self.minibatch_size
        while(iterSize < m):
            if(iterSize+self.minibatch_size < m):
                minibatches.append((X_train[iterSize:iterSize+self.minibatch_size], Y_train[iterSize:iterSize+self.minibatch_size]))
                iterSize += self.minibatch_size
            else:
                minibatches.append((X_train[iterSize:m],Y_train[iterSize:m]))
                iterSize = m
        return minibatches


    def runWithEarlyStopping(self, X_train, Y_train, X_dev, Y_dev, set_verbosity=True):

        tf.reset_default_graph()                                                            # to be able to rerun the model without overwriting tf variables
        m =int(len(X_train))
        costs_train = []
        costs_dev = []
        accuracy_train = []
        accuracy_dev = []
        yhat_each_epoch = []
        MCCs_dev = []
        # Create Placeholders of the correct shape
        X, Y, lstm_state_placeholder = self.create_placeholders()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        prediction, lstm_next_state = self.forward_propagation(X, lstm_state_placeholder)

        # Cost function: Add cost function to tensorflow graph
        cost = self.compute_cost(prediction, Y)
        #cost = self.compute_cost(prediction, Y)

        # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)
        #grads = tf.train.AdamOptimizer(learning_rate = learning_rate).compute_gradients(cost)

        # Initialize all the variables globally
        init = tf.global_variables_initializer()

        #This is for computing the test accuracy every epoch
        predict_op = tf.to_float(prediction > 0.5)

        correct_prediction = tf.equal(predict_op, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:
            sess.run(init)


          # Do the training loop
            for epoch in range(self.num_epochs):
                # 1 perche per ora ho 1 sola sequenza
                lstm_state = self.get_initial_state()
                #lstm_state = np.zeros([num_lstm_layers, 2, 1, num_lstm_units])                                                     #Ogni epoch reinizializzo stato

                minibatch_cost = 0.0
                num_minibatches = int(m / self.minibatch_size)
                if(num_minibatches == 0):
                    num_minibatches = 1
                minibatches = self.mini_batches(X_train, Y_train)


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
                    lstm_temp_state = self.get_initial_state()
                    trainCost, lstm_temp_state = sess.run([cost, lstm_next_state], feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1)), lstm_state_placeholder: lstm_temp_state})
                    costs_train.append(trainCost)
                    devCost = sess.run(cost, feed_dict={X: np.asarray(X_dev), Y: np.asarray(Y_dev).reshape((len(Y_dev), 1)), lstm_state_placeholder: lstm_temp_state})
                    costs_dev.append(devCost)


                if  epoch % 5 == 0:
                    lstm_temp_state  = self.get_initial_state()
                    yhat_train, lstm_temp_state = sess.run([prediction, lstm_next_state], feed_dict={X: np.asarray(X_train), Y: np.asarray(Y_train).reshape((len(Y_train), 1)),lstm_state_placeholder: lstm_temp_state})
                    yhat_test = sess.run(prediction, feed_dict={X: np.asarray(X_dev), Y: np.asarray(Y_dev).reshape((len(Y_dev), 1)),lstm_state_placeholder: lstm_temp_state})
                    #Acc in threshlod for train
                    thresholds = np.linspace(0, max(np.absolute(Y_train)), num=6)[:-1]
                    accs_in_threshold = list()
                    for t in thresholds:
                        correct = 0
                        n_samples_in_threshold = 0
                        for i in range(len(Y_train)):
                            if(abs(Y_train[i]) > t):
                                n_samples_in_threshold +=1
                                if( Y_train[i]*yhat_train[i] > 0):
                                    correct += 1
                        accs_in_threshold.append(correct/n_samples_in_threshold)
                    accuracy_train.append(accs_in_threshold)

                    #Acc in threshlod for test
                    thresholds = np.linspace(0, max(np.absolute(Y_dev)), num=6)[:-1]
                    accs_in_threshold = list()
                    for t in thresholds:
                        correct = 0
                        n_samples_in_threshold = 0
                        for i in range(len(Y_dev)):
                            if(abs(Y_dev[i]) > t):
                                n_samples_in_threshold += 1
                                correct += 1 if( Y_dev[i]*yhat_test[i] > 0) else 0  
                        accs_in_threshold.append(correct/n_samples_in_threshold)
                    accuracy_dev.append(accs_in_threshold)


                    # yhat_each_epoch.append(yhat)
                    # cm = confusion_matrix(Y_dev, yhat)
                    # tn, fp, fn, tp = cm.ravel()
                    # denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
                    # MCCs_dev.append(0 if denom== 0 else (tp*tn -fp*fn)/sqrt(denom) )


                if  set_verbosity and epoch % 5 == 0:
                    #print('miniCost ='+str(minibatch_cost))
                    print("Epoch "+str(epoch)+": \tTrain cost: "+str(trainCost)+" \tDev cost: "+str(devCost)+" \tTrain Accuracy: "+str(accuracy_train[-1])+" \tDev accuracy: "+str(accuracy_dev[-1]))

        # plt.figure(figsize=(20,10))
        # plt.plot(range(0,len(MCCs_dev)),MCCs_dev,'--r', label='MCCs_dev' )
        # plt.ylabel('accuracy')
        # plt.xlabel('epochs')
        # plt.title("MCC ")
        # plt.legend()
        # plt.show() 
            if(set_verbosity):
                plt.figure(figsize=(20,10))

                #colors = [name for name, _ in mcolors.cnames.items()]
                colors = ['k','b','c','r','g','y','m']

                for i in range(len(accuracy_dev)):
                    plt.plot(accuracy_train[i], linestyle='--', color=colors[i], label='accuracy_train_epoch_'+str(i))
                    plt.plot(accuracy_dev[i], color=colors[i], label='accuracy_test_epoch_'+str(i))


                plt.ylabel('accuracy')
                plt.xlabel('thresholds')
                plt.title("Learning rate =" + str(self.learning_rate))
                plt.legend()
                #plt.show()

            return (accuracy_train, accuracy_dev), yhat_each_epoch



        





# SINGLE RUN
learning_rate=0.01
batch_size=512


num_lstm_layers = 1
num_lstm_units = 100
newsTimeToMarket = 0


tickers = ['AAPL','AMZN','GOOGL','MSFT','FB','INTC','CSCO','CMCSA','NVDA','NFLX']       
for ticker in tickers:
# ticker='AAPL'
    print('\n\n\n==================== '+str(ticker)+' ==================== \n\n\n')
    train_accs = []
    test_accs = []
    MCCs = []
    MCCsReal = []
    TP = []
    TN = []
    FP = []
    FN = []
    always_yes = []


    Ttm_range = [0, 7, 14, 21, 28, 35, 70, 105, 210]

        
        # Each day: 7 hours of trading
    for newsTimeToMarket in Ttm_range:
        Data.load_data(ticker=ticker,
                       momentum_window=28,
                       newsTimeToMarket = newsTimeToMarket, X_window_average=10, set_verbosity=False)

        (X_train, Y_train), (X_test, Y_test) = Data.get_train_test_set()


        model = MyModel(num_lstm_units=num_lstm_units, num_lstm_layers=num_lstm_layers, news_per_hour = len(X_train[0]),
                        learning_rate = learning_rate, num_epochs = 30, minibatch_size = batch_size)

        (train_accuracy, test_accuracy), yhat_each_epoch = model.runWithEarlyStopping(X_train, Y_train, X_test, Y_test)
        plt.savefig('QuantileRegressionExperiments/'+ticker+'_'+str(newsTimeToMarket)+'.jpg')

        # yhat = [sign(x-0.5) for x in yhat_each_epoch[-1]]  
        # cm = confusion_matrix(Y_test, yhat)
        # tn, fp, fn, tp = cm.ravel()
        # denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        # MCCsReal.append(0 if denom== 0 else (tp*tn -fp*fn)/sqrt(denom) )
        # TP.append(tp)
        # TN.append(tn)
        # FN.append(fn)
        # FP.append(fp)

        # if(tp + fp == 0):
        #     tp = 1
        # if(tp + fn == 0):
        #     tp = 1
        # if(tn + fp == 0):
        #     tn = 1
        # if(tn + fn == 0):
        #     tn = 1

        # train_accs.append(train_accuracy[-1])
        # test_accs.append(test_accuracy[-1])
        # MCCs.append((tp*tn -fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        # b = np.sum(np.asarray(Y_test==1))/len(Y_test)
        # b = 1-b if(b<0.5) else b
        # always_yes.append(b)


    # print(ticker)
    # print('Ttm_range, '+str(Ttm_range))
    # print('y=1,'+str(ticker)+', '+str(always_yes))
    # print('test,'+str(ticker)+', '+str(test_accs))
    # print('MCC,'+str(ticker)+', '+str(MCCs))
    # print('MCC_R,'+str(ticker)+', '+str(MCCsReal))
    # print('TN,'+str(ticker)+', '+str(TN))
    # print('FP,'+str(ticker)+', '+str(FP))
    # print('FN,'+str(ticker)+', '+str(FN))
    # print('TP,'+str(ticker)+', '+str(TP))


