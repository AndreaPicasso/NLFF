%% MAKE DATASET
clear; clc; close all;

load('dataset.mat');
fprintf("Dataset created \n");
      

      

%%
momentum = 30;


[n,d] = size(X);

Z = -ones(size(Y,1),1);

for i=momentum+1:n
    Z(i)=Y(i)-Y(i-momentum);
end
Y = sign(Z);
Y(Y==0) = -1;


%X = movmean(X,[15 0],1);


X = X(momentum:end,:);
Y = Y(momentum:end,:);
[n,d] = size(X);


%Normalizzo
for j = 1:d
    mi = min(X(:,j));
    ma = max(X(:,j));
    di = ma - mi;
    if (di < 1.e-6)
        X(:,j) = 0;
    else
        X(:,j) = 2*(X(:,j)-mi)/di -1;
    end
end


fprintf("Normalization Done \n")

%% Retain last part for testing

nTest = round(.2*n);
XTest = X(end-nTest+1:end,:);
YTest = Y(end-nTest+1:end,:);


X = X(1:end-nTest,:);
Y = Y(1:end-nTest,:);
[n,d] = size(X);




 %% Make classes balanced:
% x = X(Y<0,:);
% y = Y(Y<0,:);
% 
% index = Y>0;
% count = size(x,1);
% for i=1:size(index,1)
%     if(count == 0)
%         index(i) =0;
%     end
%     if(index(i) >0)
%         count = count -1;
%     end
% end
% X = [x; X(index,:)];
% Y = [y; Y(index,:)];
% 
% [n,d] = size(X);
% fprintf("Dataset Balanced \n")
% 


%%

Xoriginal =X;
Yoriginal = Y;


%% SVM SEQUENTIAL Model Selection


nk= 8; %numero cross validation


err_best = +Inf;
train_errs = [];
test_errs = [];

H = diag(Y)*(X*X')*diag(Y);
splitting = floor(linspace(1, n, 1.6*nk));
splitting = splitting(end-nk:end);
fprintf('nv: %d \n',splitting(2)-splitting(1));

for C = logspace(-6,4,50)
    %splitto il dataset
    err = 0;
    tr_err = 0;
    baseline = 0;
    for k=1:nk
        v = splitting(k):splitting(k+1);
        
        UB = C*ones(n,1);
        % mi baster� settare questi indici a 0 per non contarare i punti nel training
        UB(v(1):end) = 0;
        
        %Balancing of training set
        index = Y(1:v(1)) > 0;
        count = sum(Y(1:v(1)) < 0);
        for i = 1:v(1)
            if(index(i) > 0)
                if(count <= 0)
                    UB(i) = 0;
                end
                count = count -1;
            end
        end
        

        [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
        
        %Training Error
        YP = X(UB~= 0 ,:)*X'*diag(Y)*alpha+b;
        YF = YP.*Y(UB~= 0);
        tr_err = tr_err+mean(YF<=0)/nk;
        
        YP = X(v,:)*X'*diag(Y)*alpha+b;
        YF = YP.*Y(v);
        %YF = H*alpha+b; %classifico tutti i punti
        err = err+mean(YF<=0)/nk;
        baseline = baseline +mean(Y(v)<=0)/nk; 
    end
    test_errs = [test_errs; err];
    train_errs = [train_errs; tr_err];
    fprintf('C: %e err: %.3f err*: %.3f base: %.3f\n', C, err, err_best, baseline);
    if(err_best > err)
        %se miglioro l'errore best aggiorno
        err_best = err;
        C_best = C;
    end
end
% valori di default sono 30% test se voglio fare model selection
% 20% test 20% validation se voglio fare error estimation
fprintf('OPT: C = %e \n', C_best );
semilogx(logspace(-9,3,50), 1-train_errs, 'b', logspace(-9,3,50), 1-test_errs, 'r',logspace(-9,3,50), 0.759*ones(size(test_errs)), 'k--');
title('SVM')
legend('train accuracy', 'test accuracy', 'base');
xlabel('C');


% %a questo punto posso rifare l'ottimizzazione con i parapetri trovati
% C=C_best;
% UB = C*ones(n,1);
% %Balancing of training set
% index = Y > 0;
% count = sum(Y < 0);
% for i = 1:size(Y,1)
%     if(index(i) > 0)
%         if(count <= 0)
%             UB(i) = 0;
%         end
%         count = count -1;
%     end
% end
% 
% H = diag(Y)*(X*X')*diag(Y);
% %[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),UB,1.e+8,1.e-4,zeros(n,1));
% [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y, UB,1e+8,1e-3,zeros(n,1));
% 
% 
% 
% 
% ns = 50000;
% XS = 2*rand(ns,d)-1;
% YS = XS*X'*diag(Y)*alpha+b;
% 
% figure; hold on; box on; grid on;
% 
% plot(XS(YS>0,1),XS(YS>0,2),'.c')
% plot(XS(YS<0,1),XS(YS<0,2),'.m')
% plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
% plot(XS(YS<-1,1),XS(YS<-1,2),'.r')
%  
% plot(X(Y==+1,1),X(Y==+1,2),'ob', 'MarkerFaceColor','b', 'MarkerSize',8, 'MarkerEdgeColor','k')
% plot(X(Y==-1,1),X(Y==-1,2),'or','MarkerFaceColor','r', 'MarkerSize',8, 'MarkerEdgeColor','k')
%  
% plot(X(alpha==C,1),X(alpha==C,2),'*g','MarkerSize',6)
% plot(X(alpha==0,1),X(alpha==0,2),'*y','MarkerSize',6)
% plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k','MarkerSize',6)

%% Error on test:
C=C_best;
UB = C*ones(n,1);
%Balancing of training set
index = Y > 0;
count = sum(Y < 0);
for i = 1:size(Y,1)
    if(index(i) > 0)
        if(count <= 0)
            UB(i) = 0;
        end
        count = count -1;
    end
end

H = diag(Y)*(X*X')*diag(Y);
%[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),UB,1.e+8,1.e-4,zeros(n,1));
[~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y, UB,1e+8,1e-3,zeros(n,1));

YP = XTest*X'*diag(Y)*alpha+b;
YF = YP.*YTest;
fprintf('Accuracy: %.3f Baseline: %.3f \n ',mean(YF>0), max(mean(YTest>0),mean(YTest<0)));
figure; hold on; box on; grid on;
plot(YTest,'--r');
plot(YP,'g');
plot(zeros(size(YP)),'--k')
title(['Test predictions, Accuracy: ' num2str(floor(mean(YF>0)*100)),'% baseline: 62%'])
legend('sign(p(t)-p(t-30))', 'prediction', '0')


%% SVM PLOT

C=C_best;
UB = C*ones(n,1);
%Balancing of training set
index = Y > 0;
count = sum(Y < 0);
for i = 1:size(Y,1)
    if(index(i) > 0)
        if(count <= 0)
            UB(i) = 0;
        end
        count = count -1;
    end
end

H = diag(Y)*(X*X')*diag(Y);
%[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),UB,1.e+8,1.e-4,zeros(n,1));
[~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y, UB,1e+8,1e-3,zeros(n,1));


ns = 50000;
XS = 2*rand(ns,d)-1;
YS =(XS*X')*diag(Y)*alpha+b;
figure; hold on; box on; grid on;
title('SVM Learned model: blue -> predict positive')
plot(XS(YS>0,1),XS(YS>0,2),'.c')
plot(XS(YS<0,1),XS(YS<0,2),'.m')
plot(X(Y==+1,1),X(Y==+1,2),'ob', 'MarkerFaceColor','b', 'MarkerSize',8, 'MarkerEdgeColor','k')
plot(X(Y==-1,1),X(Y==-1,2),'or','MarkerFaceColor','r', 'MarkerSize',8, 'MarkerEdgeColor','k')
% plot(X(Y==+1,1),X(Y==+1,2),'ob', 'MarkerFaceColor','b', 'MarkerSize',8)
% plot(X(Y==-1,1),X(Y==-1,2),'or','MarkerFaceColor','r', 'MarkerSize',8)
xlabel('NEGATIVE')
ylabel('POSITIVE')



%% Error Estimation (with model selection)
% 

% 
% 
% nt = floor(.2*n);
% nkEE = floor(n/nt);
% fprintf('fold: %d \n',nkEE);
% permTest = randperm(n);
% 
% F1Test = 0;
% baseline = 0;
% for kEE=1:nkEE
%     
%     %Retain Test part
%     XTest = Xoriginal(permTest(1+(kEE-1)*nt:(kEE)*nt),:);
%     YTest = Yoriginal(permTest(1+(kEE-1)*nt:(kEE)*nt),:);
%     X = Xoriginal;
%     X(permTest(1+(kEE-1)*nt:(kEE)*nt),:) = [];
%     Y = Yoriginal;
%     Y(permTest(1+(kEE-1)*nt:(kEE)*nt),:) = [];
%     [n,d] = size(X);
%     
%     % Model Selection:
%     nk= 100; %numero cross validation
%     F1_best = +Inf;
%     H = diag(Y)*(X*X')*diag(Y);
%     for C = logspace(-6,3,30)
%         fprintf('.');
%         %splitto il dataset
%         F1Val = 0;
%         for k=1:nk
%             i = randperm(n);
%             nv = round(.2*n);
%             UB = C*ones(n,1);
%             % mi baster� settare questi indici a 0 per non contarare i punti nel training
%             UB(i(1:nv)) = 0;
% 
%             [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
% 
%             YP = X(i(1:nv),:)*X'*diag(Y)*alpha+b;
%             YF = YP.*Y(i(1:nv));
%             F1Val = F1Val+mean(YF<=0)/nk; %considero solo quelli di validation
% 
%         end
%         %fprintf(' %e %.3f %.3f\n', C, errVal, err_best);
%         if(F1_best > F1Val)
%             F1_best = F1Val;
%             C_best = C;
%         end
%     end
%     
%     UB = C_best*ones(n,1);
%     [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
%     YP = XTest*X'*diag(Y)*alpha+b;
%     YF = YP.*YTest;
%     F1Test = F1Test+mean(YF<=0)/nkEE; %considero solo quelli di validation
%     baseline = baseline + mean(YTest >0)/nkEE;
%     
%     fprintf('\n nTrain: %d \t C = %e \t DevAcc: %.3f \t TestAcc: %.3f \t baseline: %.3f \n',n-nv, C_best , 1-F1_best, 1-mean(YF<=0), max(mean(YTest >0), mean(YTest <0)));
% 
%     
% end
% 
% fprintf('Estimated Error: %.3f \t Baseline: %.3f \n',1-F1Test, max(baseline, 1-baseline));
% 



%% Sequential Error Estimation with F1 score


nkEE = 4;
nt = 100;
nv = 100;

accuracies = [];

predictions = [];
realY = [];

F1Test = 0;
baseline = 0;
for kEE=1:nkEE
    %Offset finale rimosso
    X = Xoriginal(1:end-nt*(kEE-1),:);
    Y = Yoriginal(1:end-nt*(kEE-1),:);
    [n,d] = size(X);
    %Last part for testing
    XTest = X(end-nt+1:end,:);
    YTest = Y(end-nt+1:end,:);
    %Rimuovo anche la window, in modo che i punti di test non siano
    %relazionati con quelli di training
    X = X(1:end-nt,:);
    Y = Y(1:end-nt,:);
    [n,d] = size(X);


    % Model Selection:
    nk= 3; %numero cross validation
    F1_best = -Inf;
    
    H = diag(Y)*(X*X')*diag(Y);
    for C = logspace(-6,4,20)
        fprintf('.');
        %splitto il dataset
        F1Val = 0;
        for k=1:nk
            fprintf('.');
            fin = n-(k-1)*nv;
            v =  fin - nv + 1 : fin;
            UB = C*ones(n,1);
            UB(v(1) : end) = 0;

            [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
            YP = X(v,:)*X'*diag(Y)*alpha+b;
            TP = sum(Y(v)== 1 & sign(YP) == 1);
            TN = sum(Y(v)==-1 & sign(YP) ==-1);
            FP = sum(Y(v)==-1 & sign(YP) == 1);
            FN = sum(Y(v)== 1 & sign(YP) == -1);
            PREC = TP / (TP + FP); % perc predetti positivi correttamente rispetto a tutti quelli predetti positivi
            REC = TP / (TP + FN); % predetti positivi tra tutti quelli positivi
            if(TP + FN == 0)
                REC = 1;
                 %fprintf('%d TP:%d TN:%d FP:%d FN:%d  PREC:%f REC:%f',k, TP, TN, FP, FN, PREC, REC);
            end
            
            F1 = 2*PREC*REC / (PREC + REC);
            
            
            %YF = YP.*Y(v);
            %errVal = errVal+mean(YF<=0)/nk; %considero solo quelli di validation
            F1Val = F1Val+F1/nk; %considero solo quelli di validation

        end
        % fprintf('%d \t %e %e %.3f %.3f\n',sum(UB>0), gamma, C, F1Test, F1_best);
%         if(err_best > errVal)
%             err_best = errVal;
%             C_best = C;
%         end
        if(F1_best < F1Val)
            F1_best = F1Val;
            C_best = C;
        end
    end

    C=C_best;
    H = diag(Y)*(X*X')*diag(Y);
    [~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));

    YP = XTest*X'*diag(Y)*alpha+b;
    YF = YP.*YTest;
    
    TP = sum(YTest==1 & sign(YP) ==1);
    TN = sum(YTest==-1 & sign(YP) ==-1);
    FP = sum(YTest==-1 & sign(YP) == 1);
    FN = sum(YTest== 1 & sign(YP) == -1);
    PREC = TP / (TP + FP); % perc predetti positivi correttamente rispetto a tutti quelli predetti positivi
    REC = TP / (TP + FN); % predetti positivi tra tutti quelli positivi
    
    if(TP + FN == 0)
        REC = 1;
    end
    if(TP + FP == 0)
        PREC = TN / (TN + FN); %se non ve ne e manco uno predetto positivo, la precision la calcolo sui negativi
    end

    F1 = 2*PREC*REC / (PREC + REC);
    
    %err = err+mean(YF<=0)/nkEE; %considero solo quelli di validation
    F1Test = F1Test+F1/nkEE; %considero solo quelli di validation

    baseline = baseline + mean(YTest > 0)/nkEE;

    predictions = [YP; predictions];
    realY = [realY; YTest];


    fprintf('\n nt: %d \t C = %e \t Dev F1: %.3f \t Test Prec: %.3f Test Rec: %.3f Test F1: %.3f \t baseline:(Test: %.3f Train: %.3f) \n',n-nv, C_best , F1_best, PREC, REC,F1, max(mean(YTest >0), mean(YTest <0)), max(mean(Y >0), mean(Y <0)));


end
    
figure; hold on; box on; grid on;
plot(realY,'--r');
plot(predictions,'g');
%title(['[',num2str(),',',num2str(size(X,1)-nt*(kEE-1)),']']);
hold off;

fprintf('Estimated F1: %.3f \t Baseline: %.3f \n',F1Test, max(baseline,1-baseline));







 

%% Momentum window SELECTION
momentums = [1,3,5, 10:10:100, 120, 300, 500 ];
accuracies = [];
for momentum = momentums
    X = Xoriginal;
    Y = Yoriginal;
    [n,d] = size(X);

    Z = -ones(size(Y,1),1);

    for i=momentum+1:n
        Z(i)=Y(i)-Y(i-momentum);
    end
    Y = sign(Z);
    Y(Y==0) = -1;

    X = X(momentum:end,:);
    Y = Y(momentum:end,:);
    [n,d] = size(X);


    %Normalizzo
    for j = 1:d
        mi = min(X(:,j));
        ma = max(X(:,j));
        di = ma - mi;
        if (di < 1.e-6)
            X(:,j) = 0;
        else
            X(:,j) = 2*(X(:,j)-mi)/di -1;
        end
    end

    %dataset balanced
    x = X(Y<0,:);
    y = Y(Y<0,:);

    index = Y>0;
    count = size(x,1);
    for i=1:size(index,1)
        if(count == 0)
            index(i) =0;
        end
        if(index(i) >0)
            count = count -1;
        end
    end
    X = [x; X(index,:)];
    Y = [y; Y(index,:)];

    [n,d] = size(X);

    %Model selection
    
    nk= 40; %numero cross validation
    F1_best = +Inf;

    H = diag(Y)*(X*X')*diag(Y);
    for C = logspace(-6,3,30)
        %splitto il dataset
        F1Test = 0;
        for k=1:nk
            i = randperm(n);
            nv = round(.3*n); %30% dei dati per la validation
            UB = C*ones(n,1);
            UB(i(1:nv)) = 0;
            [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
            YP = X(i(1:nv),:)*X'*diag(Y)*alpha+b;
            YF = YP.*Y(i(1:nv));
            F1Test = F1Test+mean(YF<=0)/nk; %considero solo quelli di validation

        end
        %fprintf(' %e %.3f %.3f\n', C, err, err_best);
        if(F1_best > F1Test)
            %se miglioro l'errore best aggiorno
            F1_best = F1Test;
            C_best = C;
        end
    end
    fprintf('OPT: C = %e \n', C_best );

    %a questo punto posso rifare l'ottimizzazione con i parapetri trovati
    C=C_best;
    H = diag(Y)*(X*X')*diag(Y);
    % diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
    [~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));


    
    nk=100;
    F1Test = 0;
    for k=1:nk

        i = randperm(n);
        nv = round(.3*n); %30% dei dati per la validation
        UB = C*ones(n,1);
        % mi baster� settare questi indici a 0 per non contarare i punti nel training
        UB(i(1:nv)) = 0;

        [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

        YP = X(i(1:nv),:)*X'*diag(Y)*alpha+b;
        YF = YP.*Y(i(1:nv));
        F1Test = F1Test+mean(YF<=0)/nk; %considero solo quelli di validation
    end
    fprintf('Window: %d Accuacy: %.3f\n',momentum, 1-F1Test);
    accuracies = [accuracies; 1-F1Test];

end

% plot varying moving average
momentums = [1 3 5 10 20 30 40 50 60 70 80 90 100 120 300 500 600];
accuracyes = [0.495 0.499 0.534 0.564 0.589 0.594 0.592 0.578 0.563  0.519 0.528 0.540 0.572 0.577 0.558 0.521 0.48];
figure; box on; grid on;
semilogx(momentums, accuracyes)
xlabel('momentum window')
ylabel('accuracy')


