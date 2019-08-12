import model
import numpy as np
import load_data
import modelStateful
import matplotlib.pyplot as plt
import setBatchSize
import crossValidation

print('Loading Data...')
(x_train, y_train), (x_test, y_test) = load_data.load_data(1,30)

nfold=40
learning=[0.01,0.001,0.0001,0.00001,0.000001]
mem=[64,128]
reg=[0.001,0.05,0.1,0.5]
drop=[0,0.25,0.5]
resultsacc=list()
resultsbase=list()

for d in drop:
    for r in reg:
        for m in mem:
            for l in learning:
                acclist, baselist = crossValidation.crossValidation(x_train, y_train, nfold, l, m, r,d)
                acc=np.asarray(acclist)
                base=np.asarray(baselist)
                acc=acc.mean()
                base=base.mean()

                print('Drop:',d)
                print('Reg:',r)
                print('Mem:',m)
                print('Learning',l)
                print('AAAAAAAccuracy:',acc)
                print('BBBBBBBBBBBase:',base)
                print('------------------------------------------------------------')


'''''
        tempacc.append(acclist)
        tempbase.append(baselist)

    resultsacc.append(tempacc)
    resultsbase.append(tempbase)

for i in range(0,len(resultsacc)):
    tempa=resultsacc[0]
    tempb=resultsbase[0]
    for i in range(0,len(tempa)):

        plt.plot(tempa[i])
        plt.plot(tempb[i])
        plt.title('model accuracy :')
        plt.ylabel('accuracy')
        plt.xlabel('fold')
        plt.legend(['model', 'base'], loc='upper left')
        plt.show()
        acc=np.asarray(tempa[i])
        acc=acc.mean(acc)
        base = np.asarray(tempb[i])
        base = base.mean(base)
        print('Accuracy:',acc)
        print('Base:',base)

'''''
