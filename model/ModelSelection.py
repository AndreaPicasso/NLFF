import model
import numpy as np
import load_data
import modelStateful
import matplotlib.pyplot as plt
import setBatchSize
#memory=[64,128,256]
memory=[128]
learning=[0.000001]
#learning=[0.000001,0.00001,0.0001,0.001,0.01]
regularization=[0.01]
print('Loading Data...')
(x_train, y_train), (x_test, y_test) = load_data.load_data(0.7,30)


for mem in memory:
    for l in learning:
        for r in regularization:
            print('regularization:',r)
            print('learning:',l)
            print('memory:',mem)
            modelStateful.model(l,mem, r,0,x_train, y_train, x_test, y_test)
'''''
acctrain=list()
acctest=list()
losstrain=list()
losstest=list()

for i in range(0,5):
    history=
    acctrain.append(history.history['acc'])
    acctest.append(history.history['val_acc'])
    losstrain.append(history.history['loss'])
    losstest.append(history.history['val_loss'])


plt.plot(acctrain)
plt.show()
plt.plot(acctest)
plt.show()
plt.plot(losstrain)
plt.show()
plt.plot(losstest)
plt.show()

plt.show()
np.matrix.mean(acctrain,axis=1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss :')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


print('Data Loaded....')
for r in regularization:
    for u in lstmunits:
        for s in timesteps:
            for l in learning:
                print('timestep ', s)
                print('units ', u)
                print('regularization :',r)
                print('learning: ',l)
                model.model(l,s,u, r,x_train, y_train, x_test, y_test)
'''''