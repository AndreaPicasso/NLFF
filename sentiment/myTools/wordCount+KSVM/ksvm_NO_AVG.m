
%% MAKE DATASET
clear; clc; close all;

%tickers = {'AAPL','ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALGN', 'ALXN', 'AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};
tickers = {'AAPL'};

 

for i =1:size(tickers,2)
    X = [];
    Y = [];
    fprintf('%s \n',tickers{i});
    tickerSentiment = importSentiment(strcat('SentimentNews/',tickers{i},'.csv'));

    tickerIndexes = importIndexes(strcat('/home/simone/Desktop/NLFF/DataSetIndexesLabeled/indexes',tickers{i},'.csv'));
    initDate = max([tickerSentiment.initTime(1) tickerIndexes.data(1)]);
    finalDate = min([tickerSentiment.initTime(end) tickerIndexes.data(end)]);
    
    tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
    tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data>= initDate, :);
        
    X = [X; [tickerSentiment.CONSTRAINING  tickerSentiment.LITIGIOUS tickerSentiment.NEGATIVE tickerSentiment.POSITIVE tickerSentiment.UNCERTAINTY]];
    %X = [X; [tickerSentiment.NEGATIVE tickerSentiment.POSITIVE]];
    Y = [Y; tickerIndexes.close];       
    
       
end





fprintf("Dataset created \n");




%%
momentum = 30;

[n,d] = size(X);

Z = -ones(size(Y,1),1);

for i=momentum+1:n
    Z(i)=Y(i)-Y(i-momentum);
end

Y = sign(Z);


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





%% Make classes balanced:
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
fprintf("Dataset Balanced \n")

%% 
Xoriginal = X;
Yoriginal = Y;

%% KSVM Model Selection -> cerco il miglior C, gamma


nk= 10; %numero cross validation
err_best = +Inf;

Q = pdist2(X,X);
for gamma = logspace(-6,4,30)
    H = diag(Y)*exp(-gamma*Q)*diag(Y);
    for C = logspace(-6,4,30)
        %splitto il dataset
        err = 0;
        for k=1:nk
            i = randperm(n);
            nv = round(.3*n); %30% dei dati per la validation
            UB = C*ones(n,1);
            % mi baster� settare questi indici a 0 per non contarare i punti nel training
            UB(i(1:nv)) = 0;

            [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
            
            YP = exp(-gamma*pdist2(X(i(1:nv),:),X))*diag(Y)*alpha+b;
            YF = YP.*Y(i(1:nv));
            %YF = H*alpha+b; %classifico tutti i punti
            err = err+mean(YF<=0)/nk; %considero solo quelli di validation
                        
        end
        fprintf('%e %e %.3f %.3f\n', gamma, C, err, err_best);
        if(err_best > err)
            %se miglioro l'errore best aggiorno
            err_best = err;
            C_best = C;
            gamma_best = gamma;
        end
    end
end     
% valori di default sono 30% test se voglio fare model selection
% 20% test 20% validation se voglio fare error estimation
fprintf('OPT: C = %e gamma = %e\n', C_best, gamma_best);

%a questo punto posso rifare l'ottimizzazione con i parapetri trovati
C=C_best;
gamma=gamma_best;
%H = diag(Y)*(X*X')*diag(Y); sostinuiamo con il kernel
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
% diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));



ns = 10000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
figure; hold on; box on; grid on;

plot(XS(YS>0,3),XS(YS>0,4),'.c')
plot(XS(YS<0,3),XS(YS<0,4),'.m')
plot(XS(YS>+1,3),XS(YS>+1,4),'.b')
plot(XS(YS<-1,3),XS(YS<-1,4),'.r')
 
plot(X(Y==+1,3),X(Y==+1,4),'ob', 'MarkerFaceColor','b', 'MarkerSize',8)
plot(X(Y==-1,3),X(Y==-1,4),'or','MarkerFaceColor','r', 'MarkerSize',8)
 
plot(X(alpha==C,3),X(alpha==C,4),'*g','MarkerSize',6)
plot(X(alpha==0,3),X(alpha==0,4),'*y','MarkerSize',6)
plot(X(alpha>0&alpha<C,3),X(alpha>0&alpha<C,4),'*k','MarkerSize',6)
 
YP = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
YP = sign(YP);
tot = sum(YS == YP)/size(YS, 1);

%% KERNEL PLOT

C=C_best; % 8.531679e+01
gamma=gamma_best; % 1.373824e-02

ns = 50000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
figure; hold on; box on; grid on;
title('Learned model: blue -> predict positive: accuracy: 0.60')
plot(XS(YS>0,3),XS(YS>0,4),'.c')
plot(XS(YS<0,3),XS(YS<0,4),'.m')
xlabel('NEGATIVE')
ylabel('POSITIVE')

%% Error Estimation (with model selection)
% 



nt = floor(.2*n);
nkEE = floor(n/nt);
fprintf('fold: %d \n',nkEE);
permTest = randperm(n);

err = 0;
baseline = 0;
for kEE=1:nkEE
    
    %Retain Test part
    XTest = Xoriginal(permTest(1+(kEE-1)*nt:(kEE)*nt),:);
    YTest = Yoriginal(permTest(1+(kEE-1)*nt:(kEE)*nt),:);
    X = Xoriginal;
    X(permTest(1+(kEE-1)*nt:(kEE)*nt),:) = [];
    Y = Yoriginal;
    Y(permTest(1+(kEE-1)*nt:(kEE)*nt),:) = [];
    [n,d] = size(X);
    
    % Model Selection:
    nk= 30; %numero cross validation
    err_best = +Inf;
    Q = pdist2(X,X);
    for gamma = logspace(-6,4,20)
        fprintf('.');
        H = diag(Y)*exp(-gamma*Q)*diag(Y);
        for C = logspace(-6,4,20)
            %splitto il dataset
            errVal = 0;
            for k=1:nk
                i = randperm(n);
                nv = round(.2*n); %30% dei dati per la validation
                UB = C*ones(n,1);
                % mi baster� settare questi indici a 0 per non contarare i punti nel training
                UB(i(1:nv)) = 0;

                [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

                YP = exp(-gamma*pdist2(X(i(1:nv),:),X))*diag(Y)*alpha+b;
                YF = YP.*Y(i(1:nv));
                %YF = H*alpha+b; %classifico tutti i punti
                errVal = errVal+mean(YF<=0)/nk; %considero solo quelli di validation

            end
            %fprintf('%e %e %.3f %.3f\n', gamma, C, errVal, err_best);
            if(err_best > errVal)
                %se miglioro l'errore best aggiorno
                err_best = errVal;
                C_best = C;
                gamma_best = gamma;
            end
        end
    end     
    
    UB = C_best*ones(n,1);
    H = diag(Y)*exp(-gamma_best*Q)*diag(Y);
    
    [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
    YP = exp(-gamma_best*pdist2(XTest,X))*diag(Y)*alpha+b;

    YF = YP.*YTest;
    err = err+mean(YF<=0)/nkEE; %considero solo quelli di validation
    baseline = baseline + mean(YTest >0)/nkEE;
    
    fprintf('\n nTrain: %d \t (C = %e, gamma = %e) \t DevAcc: %.3f \t TestAcc: %.3f \t baseline: %.3f \n',n-nv, C_best ,gamma_best, 1-err_best, 1-mean(YF<=0), max(mean(YTest >0), mean(YTest <0)));

    
end

fprintf('Estimated Error: %.3f \t Baseline: %.3f \n',1-err, max(baseline, 1-baseline));


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
    
    Q = pdist2(X,X);
    for gamma = logspace(-6,4,20)
        fprintf('.');
        H = diag(Y)*exp(-gamma*Q)*diag(Y);
        for C = logspace(-6,4,20)
            %splitto il dataset
            F1Val = 0;
            for k=1:nk
                fprintf('.');
                fin = n-(k-1)*nv;
                v =  fin - nv + 1 : fin;
                UB = C*ones(n,1);
                UB(v(1) : end) = 0;

                [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
                YP = exp(-gamma*pdist2(X(v,:),X))*diag(Y)*alpha+b;
                
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
                gamma_best = gamma;
            end
        end
    end

    gamma = gamma_best;
    C=C_best;
    H = diag(Y)*exp(-gamma*Q)*diag(Y);
    [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,C*ones(n,1),1e+8,1e-3,zeros(n,1));

    YP = exp(-gamma*pdist2(XTest,X))*diag(Y)*alpha+b;
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

