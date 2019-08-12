import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tensorflow as tf




# . LOAD DATA DEGLI SKIP THROUGH VECTOR E DEL MERCATO
# .
# . CON UN MODELLO LSTM (Recurrent) non si puo fare shuffling altrimenti si perde informazione sulla sequenza
# .                                        CREDO!!!!
# .
# .



X_COLUMN_NAMES = ['PUBLICATION_DATE', 'EMBEDDING']
#Y_COLUMN_NAMES = ['DATE', 'close', 'high','low', 'close','volume','close_12_ema','close_26_ema','macd','macds','macdh','macd','macds', 'boll_ub', 'boll_lb', 'rsi_6','rsi_12','vr_6_sma','wr_10','wr_6']

X_path = '/home/andrea/Desktop/AAPL_EMBEDDING.json'
Y_path = '/home/andrea/Desktop/NLFF/DataSetIndexes/indexesAAPL.csv'

def sign(x):
    if x >= 0:
        return 1
    elif x < 0:
        #return -1
        return 0



def load_data(test_percentage=0.7, momentum_window=10, news_per_hour = 10, newsTimeToMarket = 0):

    print('Reading dataset...')
    x = pd.read_json(X_path)
    #cambio l'ordine dalla piu vecchia alla piu recente
    print('Ordering dataset...')
    x = x.sort_values(by=['PUBLICATION_DATE'])
    x = x.reset_index(drop=True)
    print(x.head())

    for i, row in x.iterrows():
        x.at[i,'PUBLICATION_DATE'] =datetime.strptime(x['PUBLICATION_DATE'][i], '%Y-%m-%d %H:%M:%S +%f') + timedelta(minutes=newsTimeToMarket)
        x.at[i, 'EMBEDDING']= x['EMBEDDING'][i][0]

    y = pd.read_csv(Y_path)
    y = y.rename(index=str, columns={"Unnamed: 0": "DATE"})

    #PER ORA SCARTO GLI INDICI, POI SARA' DA METTERLI DENTRO X
    #y = y['DATE', 'close']
    for i, row in y.iterrows():
        y['DATE'].at[i] = datetime.strptime(y['DATE'][i], '%Y-%m-%d %H:%M:%S') 

    z = list()
    print('y(t) - y(t-1) ...')

    #calcolo differenza price(t) - price(t-window)
    for i in range(0,momentum_window):
        z.append(-1) #Il primo poi tanto verra elliminato perche non sara in range
    for i in range(momentum_window,y.shape[0]):
        z.append(sign(y['close'][i] - y['close'][i-momentum_window]))
    y['close'] = z

    X = list()
    Y = list()
    
    print('Alligning dataset and constructing cube..')

    initDate = max(y['DATE'][0], x['PUBLICATION_DATE'][0])
    finalDate = min(y['DATE'][len(y)-1], x['PUBLICATION_DATE'][len(x)-1])
    i = 0
    j = 0

    # ALLINEAMENTO INIZIO
    while(y['DATE'][j] < initDate):
        j+=1
    while(x['PUBLICATION_DATE'][i] < initDate):
        i+=1

    while(x['PUBLICATION_DATE'][i] < finalDate and y['DATE'][j] < finalDate ):
        timeSlotX = list()
        while(i<len(x)-1 and y['DATE'][j] > x['PUBLICATION_DATE'][i]):
            timeSlotX.append(x['EMBEDDING'][i]) 
            i+=1
            if(i%1000 == 0):
                print(str(i)+ '/' + str(x.shape[0]))


        # Da len(timeslot) dobbiamo ricondurci ad avere news_per_hour numero di news
        # Random sampling se sono troppe:
        if(len(timeSlotX) > news_per_hour):
            #timeSlotX = np.random.choice(timeSlotX, news_per_hour, replace=False)
            selectedIndexes = np.random.choice(range(0,len(timeSlotX)-1), news_per_hour, replace=False).tolist()
            timeSlotX =  [timeSlotX[index] for index in selectedIndexes]
            
        # Replicazione news se sono troppo poche
        else:
            if(len(timeSlotX) < news_per_hour):
                index = 0
                #Se non e presente manco una news riempi di zeri
                if(len(timeSlotX) == 0):
                    timeSlotX.append([0] * 2400)
                    
                numNews = len(timeSlotX)

                while(len(timeSlotX) < news_per_hour):
                    timeSlotX.append(timeSlotX[index%numNews])
                # prevSlot = len(X)-1

                # while(len(timeSlotX) < news_per_hour and prevSlot >= 0):
                #     prevNews = len(X[prevSlot])-1
                #     while(len(timeSlotX) < news_per_hour and prevNews >=0):
                #         timeSlotX.append( X[prevSlot][prevNews])
                #         prevNews -= 1
                #     prevSlot -=1
                # #Se non ci sono news precedenti a sufficienza riempi con embedding di zeri
                # if(len(timeSlotX) < news_per_hour):
                #     while(len(timeSlotX) < news_per_hour):
                #         timeSlotX.append([0] * 2400)





        X.append(timeSlotX)   
        Y.append(y['close'][j])
        j+=1



    idx_split = math.floor(len(X)*(1-test_percentage))
    train_x = X[:idx_split]
    test_x = X[idx_split:]
    train_y = Y[:idx_split]
    test_y = Y[idx_split:]
    # test_x = tf.convert_to_tensor(np.asarray(test_x), dtype=tf.float32)
    # train_x = tf.convert_to_tensor(np.asarray(train_x), dtype=tf.float32)
    # train_y = tf.convert_to_tensor(np.asarray(train_y), dtype=tf.float32)
    # test_y = tf.convert_to_tensor(np.asarray(test_y), dtype=tf.float32)



    return (train_x, train_y), (test_x, test_y)






















# def train_input_fn(features, labels, batch_size):
#     """An input function for training"""
#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

#     # Shuffle, repeat, and batch the examples.
#     # /!\ IO NON POSSO FARE SHUFFLING COSI SEMPLICEMENTE
#     dataset = dataset.shuffle(1000).repeat().batch(batch_size)

#     # Return the dataset.
#     return dataset






# def eval_input_fn(features, labels, batch_size):
#     """An input function for evaluation or prediction"""
#     features=dict(features)
#     if labels is None:
#         # No labels, use only features.
#         inputs = features
#     else:
#         inputs = (features, labels)

#     # Convert the inputs to a Dataset.
#     dataset = tf.data.Dataset.from_tensor_slices(inputs)

#     # Batch the examples
#     assert batch_size is not None, "batch_size must not be None"
#     dataset = dataset.batch(batch_size)

#     # Return the dataset.
#     return dataset


# # The remainder of this file contains a simple example of a csv parser,
# #     implemented using a the `Dataset` class.

# # `tf.parse_csv` sets the types of the outputs to match the examples given in
# #     the `record_defaults` argument.
# CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

# def _parse_line(line):
#     # Decode the line into its fields
#     fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

#     # Pack the result into a dictionary
#     features = dict(zip(CSV_COLUMN_NAMES, fields))

#     # Separate the label from the features
#     label = features.pop('Species')

#     return features, label


# def csv_input_fn(csv_path, batch_size):
#     # Create a dataset containing the text lines.
#     dataset = tf.data.TextLineDataset(csv_path).skip(1)

#     # Parse each line.
#     dataset = dataset.map(_parse_line)

#     # Shuffle, repeat, and batch the examples.
#     dataset = dataset.shuffle(1000).repeat().batch(batch_size)

#     # Return the dataset.
#     return dataset
