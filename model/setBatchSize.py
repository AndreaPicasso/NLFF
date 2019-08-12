
def setBatchSize(x_train,y_train, x_test,y_test, size):
    toRemove=(len(x_train)%size)
    newxTrain=list()
    newyTrain = list()
    count=1
    #print(len(x_train))
    for i in range(0,len(x_train)):
        if(count>toRemove):
            newxTrain.append(x_train[i])
            newyTrain.append(y_train[i])
        count+=1
    #print(len(newxTrain))

    toRemove = (len(x_test) % size)
    newxTest=list()
    newyTest=list()
    count = 1
    #print(len(x_test))
    for i in range(0,len(x_test)):
        if((count < len(x_test)-toRemove+1)):
            newxTest.append(x_test[i])
            newyTest.append(y_test[i])
        count+=1

    #print(len(newxTest))
    return newxTrain,newyTrain,newxTest,newyTest
