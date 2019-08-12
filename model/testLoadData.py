import load_data3
import load_data

(x_train1, y_train1), (x_test1, y_test1) = load_data.load_data(0.7,10)
(x_train3, y_train3), (x_test3, y_test3) = load_data3.load_data(0.3,10)
print(len(x_train1))
print(len(x_train3))

for i in range(0,len(x_train1)):
    if(x_train1[i]==x_train3[i]):
        print(0)
    else:
        print(0000000000000)