
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

    tickerIndexes = importIndexes(strcat('/home/simone/Desktop/NLFF/DataSetIndexesLabeled/indexes',tickers{i},'.csv'));
    initDate = max([tickerSentiment.initTime(1) tickerIndexes.data(1)]);
    finalDate = min([tickerSentiment.initTime(end) tickerIndexes.data(end)]);
    
    tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
    tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data>= initDate, :);
        
    X = [X; [tickerSentiment.CONSTRAINING  tickerSentiment.LITIGIOUS tickerSentiment.NEGATIVE tickerSentiment.POSITIVE tickerSentiment.UNCERTAINTY]];
    Y = [Y; tickerIndexes.close];       
    
       
end

      





%%
momentum = 30;

[n,d] = size(X);

Z = -ones(size(Y,1),1);

for i=momentum+1:n
    Z(i)=Y(i)-Y(i-momentum);
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
        X(:,j) = 2*(X(:,j)-mi)/di -1;
    end
end

%figure; hold on; box on; grid on;
%plot(X(:,1),Z,'ob')
Xoriginal = X;

fprintf("Normalization Done \n")



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
            %YXoriginal = X;F = H*alpha+b; %classifico tutti i punti
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

%%


ns = 100000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
figure; hold on; box on; grid on;
title('Learned model: blue -> predict positive: accuracy: 0.60')
plot(XS(YS>0,3),XS(YS>0,4),'.c')
plot(XS(YS<0,3),XS(YS<0,4),'.m')
xlabel('NEGATIVE')
ylabel('POSITIVE')


%% Model Selection with window -> moving average sulla X
% invece che [#pos #neg #con .. ] -> up/down
% proviamo a prendere la moving average (nel tempo) dei vari articoli
% vediamo quali sono i risultati per ogni moving average


nk= 10; %numero cross validation

best = [];
for mov_avg_window = 1:30:700
    err_best = +Inf;
    X = movmean(Xoriginal,[mov_avg_window 0],1);
    Q = pdist2(X,X);
    for gamma = logspace(-6,4,10)
        H = diag(Y)*exp(-gamma*Q)*diag(Y);
        for C = logspace(-6,4,10)
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
            %fprintf('%e %e %.3f %.3f\n', gamma, C, err, err_best);
            if(err_best > err)
                %se miglioro l'errore best aggiorno
                fprintf('%e %e %.3f %.3f\n', gamma, C, err, err_best);
                err_best = err;
                C_best = C;
                gamma_best = gamma;
            end
        end
    end
    fprintf('-------------------------- Moving average: %d ErrBest: %.3f (%e, %e)\n', mov_avg_window, err_best, gamma_best, C_best);
    best = [best; WIN_best, gamma_best, C_best, err_best];
end


% fprintf('OPT: WIN = %d C = %e gamma = %e\n',WIN_best, C_best, gamma_best);


%% 
%a questo punto posso rifare l'ottimizzazione con i parapetri trovati
C=C_best;
gamma=gamma_best;
%H = diag(Y)*(X*X')*diag(Y); sostinuiamo con il kernel
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);ns = 100000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
figure; hold on; box on; grid on;
title('Learned model: blue -> predict positive: accuracy: 0.60')
plot(XS(YS>0,3),XS(YS>0,4),'.c')
plot(XS(YS<0,3),XS(YS<0,4),'.m')
xlabel('NEGATIVE')
ylabel('POSITIVE')
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

% WIN_best, gamma_best, C_best, err_best

% [1,0.359381366380463,4.64158883361278,0.349504504504505;
% 11,0.00215443469003188,10000,0.145315315315315;
% 21,0.359381366380463,10000,0.0606306306306307;
% 31,4.64158883361278,59.9484250318941,0.0545495495495496;
% 41,0.0278255940220713,10000,0.0515315315315316;
% 51,4.64158883361278,774.263682681128,0.0453153153153153;
% 61,0.0278255940220713,774.263682681128,0.0421171171171172]


%% Reproduce best result for WINDOW = 30

mov_avg_window = 31;

X = movmean(Xoriginal,[mov_avg_window 0],1);
gamma = 4.641589e+00;
C = 5.994843e+01;
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
% diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));

ns = 10000;

XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;


varName = {'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'};
for i = 1:5
    for j = 1:5
        figure; hold on; box on; grid on;
        title('Learned model: blue -> predict positive: accuracy: 0.60')
        plot(XS(YS>0,i),XS(YS>0,j),'.c')
        plot(XS(YS<0,i),XS(YS<0,j),'.m')
        xlabel(varName{i})
        ylabel(varName{j})
    end
end



%% cross val over best model

mov_avg_window = 30;

X = movmean(Xoriginal,[mov_avg_window 0],1);
gamma = 4.641589e+00;
C = 5.994843e+01;


H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
err = 0;
for k=1:nk
    
    i = randperm(n);
    nv = round(.3*n); %30% dei dati per la validation
    UB = C*ones(n,1);
    % mi baster� settare questi indici a 0 per non contarare i punti nel training
    UB(i(1:nv)) = 0;

    [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

    YP = exp(-gamma*pdist2(X(i(1:nv),:),X))*diag(Y)*alpha+b;
    figure; hold on; box on; grid on;
    %plot(YP, '--r*')
    %plot(Y(i(1:nv)),':bo')
    hold off;
    YF = YP.*Y(i(1:nv));
    plot(YF);
    %YF = H*alpha+b; %classifico tutti i punti
    err = err+mean(YF<=0)/nk; %considero solo quelli di validation
end
fprintf('Error: %.3f\n',err);



%% X averaged plot

%
% LA FINESTRA MIGLIORE SEMBRA ESSERE 30
% SENZA AVERAGING IN PRATICA STIAMO DANDO SOLO NOISE
% GIA CON WIN = 10 LA SITUAZIONE MIGLIORA MOLTISSIMO
%
%


colors = {'-b','-g','-m','-k','-y','-p','-c'};
varName = {'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'};
i = 1;
mov_avg_window=1;
step = 100;



for mov_avg_window = [1,10, 30, 60, 90, 120, 750]
    figure; hold on; box on; grid on;
    plot(1:size(X,1), Y,'--r')
    title(mov_avg_window);
    for varNum = 1:5
        X = movmean(Xoriginal,[mov_avg_window 0],1);
        plot(1:size(X,1), X(:,varNum), colors{varNum});
    end
    legend('Y', 'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY');
end







