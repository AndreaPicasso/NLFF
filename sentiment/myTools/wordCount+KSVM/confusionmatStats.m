function stats = confusionmatStats(confMatr)

    TN = confMatr(1,1);
    TP = confMatr(2,2);
    FN = confMatr(2,1);
    FP = confMatr(1,2);
    
    
    field2 = 'accuracy';  value2 = (TP + TN)/(TP + FP + FN + TN);
    field3 = 'sensitivity';  value3 = TP / (TP + FN);
    field4 = 'specificity';  value4 = TN / (FP + TN);
    field5 = 'precision';  value5 = TP / (TP + FP);
    field6 = 'recall';  value6 =  TP / (TP + FN);
    field7 = 'Fscore';  value7 = 2*TP /(2*TP + FP + FN);
    stats = struct(field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7);
