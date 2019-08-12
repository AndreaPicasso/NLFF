import model
import numpy as np
import load_data
import modelStateful
import matplotlib.pyplot as plt
import setBatchSize
import crossValidationFixWindow

print('Loading Data...')
(x_train, y_train), (x_test, y_test) = load_data.load_data(1,30)

nfold=10
learning=[0.000001]
mem=[1024,2048]
reg=[0.001,0.05,0.1,0.5]
drop=[0,0.25,0.5]
resultsacc=list()
resultsbase=list()

for d in drop:
    for r in reg:
        for m in mem:
            for l in learning:
                acclist, baselist = crossValidationFixWindow.crossValidation(x_train, y_train, nfold, l, m, r,d,800)
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
