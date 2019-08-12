
%% MAKE DATASET
clear; clc; close all;

%tickers = {'AAPL','ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALGN', 'ALXN', 'AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};
tickers = {'AAPL'};
 X = [];
 Y = [];
 j = 1;
 
for i =1:size(tickers,2)
    fprintf('%s \n',tickers{i});
    tickerSentiment = importSentiment(strcat('SentimentNews/',tickers{i},'.csv'));

    tickerIndexes = importIndexes(strcat('/home/simone/Scrivania/NLFF/DataSetIndexes/indexes',tickers{i},'.csv'));
    initDate = max([tickerSentiment.initTime(1) tickerIndexes.data(1)]);
    finalDate = min([tickerSentiment.initTime(end) tickerIndexes.data(end)]);
    
    tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
    tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data>= initDate, :);
        
    X = [X; [tickerSentiment.CONSTRAINING  tickerSentiment.LITIGIOUS tickerSentiment.NEGATIVE tickerSentiment.POSITIVE tickerSentiment.UNCERTAINTY]];
    Y = [Y; tickerIndexes.open];       
    
       
end
fprintf("Dataset created \n");
      
      

%% Dataset Reduction
% per ora non consideriamo i link temporali ma semplicemente al tempo t
% prediciamo l'andamento del mercato con il sentiment t per il tempo t
% perciò possiamo semplicemente fare random sampling

n = 10000;

perm = randperm(size(X,1));
X = X(perm(1:n),:);
Y= Y(perm(1:n),:);







%%

[n,d] = size(X);

Z = -ones(size(Y,1),1);

for i=2:n
    Z(i)=Y(i)-Y(i-1);
%     Y(i)=sign(Z(i));
%     if(Y(i)==0)
%         Y(i)=1;
%         fprintf('equals');
%     end
end
Y = sign(Z);
Y(Y==0) = -1;


%Normalizzo
for j = 1:d
    mi = min(X(:,j));
    ma = max(X(:,j));
    di = ma - mi;
    if (di < 1.e-6)
        X(:,j) = 0;
    else
        X(:,j) = 2*(X(:,j)-mi)/di-1;
    end
end

%figure; hold on; box on; grid on;
%plot(X(:,1),Z,'ob')

fprintf("Normalization Done \n")


%% divisione train e test


perm = randperm(size(X,1));
XS = X(perm(1:floor(n/3)),:);
YS = Y(perm(1:floor(n/3)),:);
X = X(perm(ceil(n/3):end),:);
Y = Y(perm(ceil(n/3):end),:);

n = n -floor(n/3);



%BEST ERROR: 0.448
%OPT: C = 1.887392e+02 gamma = 4.175319e+02


C=1.000000e+05;

%gamma=15;
gamma = 2.395027e-04;
%H = diag(Y)*(X*X')*diag(Y); sostinuiamo con il kernel
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
% diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n,1),1e+8,1e-6,zeros(n,1));
ns = 10000;


% TEST
YP = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
YP = sign(YP);
tot = sum(YS == YP)/size(YS, 1)



%% Model Selection -> cerco il miglior C, gamma


nk= 100; %numero cross validation
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
 
plot(X(Y==+1,3),X(Y==+1,4),'ob')
plot(X(Y==-1,3),X(Y==-1,4),'or')
 
plot(X(alpha==C,3),X(alpha==C,4),'*g','MarkerSize',8)
plot(X(alpha==0,3),X(alpha==0,4),'*y','MarkerSize',8)
plot(X(alpha>0&alpha<C,3),X(alpha>0&alpha<C,4),'*k','MarkerSize',8)

YP = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
YP = sign(YP);
tot = sum(YS == YP)/size(YS, 1);



%% Error Estimation

nK = 1000;

tot = 0;

for i= 1:nK
    fprintf("%d \n", i);
    perm = randperm(size(X,1));
    XS = X(perm(1:floor(n/4)),:);
    YS = Y(perm(1:floor(n/4)),:);
    XT = X(perm(ceil(n/4):end),:);
    YT = Y(perm(ceil(n/4):end),:);

    nT = n -floor(n/4);



    %BEST ERROR: 0.346
    % OPT: C = 1.000000e+05 gamma = 2.395027e-04

    C=1.000000e+05;
    gamma = 2.395027e-04;

    H = diag(YT)*exp(-gamma*pdist2(XT,XT))*diag(YT);

    [~,~,alpha,b] = SMO2_ab(nT,H,-ones(nT,1),YT,zeros(nT,1),C*ones(nT,1),1e+8,1e-6,zeros(nT,1));
    ns = 10000;


    % TEST
    YP = exp(-gamma*pdist2(XS,XT))*diag(YT)*alpha+b;
    YP = sign(YP);
    tot = tot + mean(YS == YP);

end

tot = tot / nK;
fprintf("Estimated accuracy: %e \n", tot);















