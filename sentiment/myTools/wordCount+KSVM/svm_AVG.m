
%% MAKE DATASET
clear; clc; close all;

%tickers = {'AAPL','ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALGN', 'ALXN', 'AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};
tickers = {'AMZN'};
 X = [];
 Y = [];
 j = 1;

 

for i =1:size(tickers,2)
    fprintf('%s \n',tickers{i});
    tickerSentiment = importSentiment(strcat('SentimentNews2/',tickers{i},'.csv'));

    tickerIndexes = importIndexes(strcat('/home/simone/Desktop/NLFF/indexes/indexes',tickers{i},'.csv'));
    initDate = max([tickerSentiment.initTime(1) tickerIndexes.date(1)]);
    finalDate = min([tickerSentiment.initTime(end) tickerIndexes.date(end)]);
    
    tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
    tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.date<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.date>= initDate, :);
        
   % X = [X; [tickerSentiment.CONSTRAINING  tickerSentiment.LITIGIOUS tickerSentiment.NEGATIVE tickerSentiment.POSITIVE tickerSentiment.UNCERTAINTY]];
    X = [X; [tickerSentiment.NEGATIVE tickerSentiment.POSITIVE]];
    Y = [Y; tickerIndexes.close];       
    
       
end

fprintf("Dataset created \n");
      



% -- NORMALIZATION & MOMENTUM

momentum = 30;
[n,d] = size(X);

Z = -ones(size(Y,1),1);

for i=momentum+1:n
    Z(i)=Y(i)-Y(i-momentum);
end
Y = sign(Z);
Y(Y==0) = -1;
% Rimuovo prima parte senza momentum window

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


fprintf("Normalization & Momentum Done \n")


%% Retain test set:

nTest = round(.2*n);
% XTest = X(end-nTest+1:end,:);
% YTest = Y(end-nTest+1:end,:);
Xall = X;
Yall = Y;


X = X(1:end-nTest,:);
Y = Y(1:end-nTest,:);
[n,d] = size(X);


Xoriginal = X;
Yoriginal = Y;


%% SVM SEQUENTIAL Model Selection


nk= 5; %numero cross validation
best_F1 = 0;
windows = [1 3 5 10 15 20 25 30 35 40 45 50 100 120];
splitting = floor(linspace(1, n, 1.6*nk));
splitting = splitting(end-nk:end);
fprintf('nv: %d \n',splitting(2)-splitting(1));

for mov_avg_window = windows
    X = movmean(Xoriginal,[mov_avg_window 0],1);
    Y = Yoriginal;
    H = diag(Y)*(X*X')*diag(Y);
    for C = logspace(-6,3,30)
        %splitto il dataset
        confMatr = zeros(2,2);
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
            fprintf('nt: %d ',sum(UB>0))

            [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

            YP = X(v,:)*X'*diag(Y)*alpha+b;
            YF = YP.*Y(v);
            %YF = H*alpha+b; %classifico tutti i punti
            
            confMatr = confMatr + confusionmat(Y(v), sign(YP),'Order',[-1 1]);
            
            
        end
        stats = confusionmatStats(confMatr);
        % stats = confusionmatStats(group,grouphat);
        fprintf('MW: %d C: %e PR: %.3f REC: %.3f F1: %.3f best: %.3f\n',mov_avg_window, C,stats.precision,stats.recall, stats.Fscore, best_F1);
        if(best_F1 <  stats.Fscore)
            %se miglioro l'errore best aggiorno
            win_best = mov_avg_window;
            best_F1 = stats.Fscore;
            C_best = C;
        end
    end
end

X = movmean(Xall,[win_best 0],1);
XTest = X(end-nTest+1:end,:);
YTest = Yall(end-nTest+1:end,:);
X = X(1:end-nTest,:);
Y = Yall(1:end-nTest,:);
[n,d] = size(X);

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
fprintf('Accuracy: %.3f Baseline: %.3f (C = %.3f Window = %d) \n ',mean(YF>0), mean(YTest>0), C_best, win_best);
figure; hold on; box on; grid on;
plot(YTest,'--r');
plot(YP,'g');
plot(zeros(size(YP)),'--k')
title([' Test predictions, Window: ',num2str(win_best) ,' Accuracy: ' num2str(floor(mean(YF>0)*100)),'% baseline: 62%'])
legend('sign(p(t)-p(t-30))', 'prediction', '0')
confusionmat(YTest, sign(YP),'Order',[-1 1])

% valori di default sono 30% test se voglio fare model selection
% 20% test 20% validation se voglio fare error estimation



%% SVM SEQUENTIAL Model Selection


nk= 5; %numero cross validation
err_best = +Inf;
windows = [1 3 5 10 15 20 25 30 35 40 45 50 55 60 70 80 90 100 120];
splitting = floor(linspace(1, n, 1.6*nk));
splitting = splitting(end-nk:end);
fprintf('nv: %d \n',splitting(2)-splitting(1));

for mov_avg_window = windows
    X = movmean(Xoriginal,[mov_avg_window 0],1);
    Y = Yoriginal;
    H = diag(Y)*(X*X')*diag(Y);
    for C = logspace(-6,3,30)
        %splitto il dataset
        err = 0;
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
            fprintf('nt: %d ',sum(UB>0))

            [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

            YP = X(v,:)*X'*diag(Y)*alpha+b;
            YF = YP.*Y(v);
            %YF = H*alpha+b; %classifico tutti i punti
            err = err+mean(YF<=0)/nk;
        end
        fprintf('%d %e %.3f %.3f\n',mov_avg_window, C, err, err_best);
        if(err_best > err)
            %se miglioro l'errore best aggiorno
            win_best = mov_avg_window;
            err_best = err;
            C_best = C;
        end
    end
end

X = movmean(Xall,[win_best 0],1);
XTest = X(end-nTest+1:end,:);
YTest = Yall(end-nTest+1:end,:);
X = X(1:end-nTest,:);
Y = Yall(1:end-nTest,:);
[n,d] = size(X);

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
fprintf('Accuracy: %.3f Baseline: %.3f (C = %.3f Window = %d) \n ',mean(YF>0), mean(YTest>0), C_best, win_best);
figure; hold on; box on; grid on;
plot(YTest,'--r');
plot(YP,'g');
plot(zeros(size(YP)),'--k')
title([' Test predictions, Window: ',num2str(win_best) ,' Accuracy: ' num2str(floor(mean(YF>0)*100)),'% baseline: 62%'])
legend('sign(p(t)-p(t-30))', 'prediction', '0')
confusionmat(YTest, sign(YP),'Order',[-1 1])

% valori di default sono 30% test se voglio fare model selection
% 20% test 20% validation se voglio fare error estimation




%% TEST ERROR for each averaging


nkEE = 4;
nt = 100;
nv = 50;

accuracies = [];
windows = [1 3 5 10 15 20 25 30 35 40 45 50 55 60 70 80 90 100 100 120];
numW =1;
figure;
for mov_avg_window = windows
    fprintf('Window: %d \n',mov_avg_window);
    Xavg = movmean(Xoriginal,[mov_avg_window 0],1);
    
    predictions = [];
    realY = [];
    
    err = 0;
    baseline = 0;
    for kEE=1:nkEE
        %Offset finale rimosso
        X = Xavg(1:end-nt*(kEE-1),:);
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
        err_best = +Inf;
        fprintf('.');
        H = diag(Y)*(X*X')*diag(Y);
        for C = logspace(-6,4,20)
            %splitto il dataset
            errVal = 0;

            for k=1:nk
                fin = min(n-mov_avg_window, n-(k-1)*nv );
                v =  fin - nv + 1 : fin;
                UB = C*ones(n,1);
                UB(v(1) : end) = 0;

                [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
                YP = X(v,:)*X'*diag(Y)*alpha+b;
                YF = YP.*Y(v);
                errVal = errVal+mean(YF<=0)/nk; %considero solo quelli di validation
            end
            %fprintf('%d \t %e %e %.3f %.3f\n',sum(UB>0), gamma, C, err, err_best);
            if(err_best > errVal)
                best_win = mov_avg_window;
                err_best = errVal;
                C_best = C;
            end
        end

        C=C_best;
        H = diag(Y)*(X*X')*diag(Y);
        [~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));
        
        YP = XTest*X'*diag(Y)*alpha+b;
        YF = YP.*YTest;
        err = err+mean(YF<=0)/nkEE; %considero solo quelli di validation
        baseline = baseline + mean(YTest > 0)/nkEE;
        
        predictions = [YP; predictions];
        realY = [realY; YTest];
        
%         subplot(10,nkEE, nkEE -(kEE-1) + (nkEE)*numW); hold on; box on; grid on;
%         plot(YTest,'--r');
%         plot(YP,'g');
%         %title(['[',num2str(),',',num2str(size(X,1)-nt*(kEE-1)),']']);
%         hold off;

        fprintf('\n nt: %d \t C = %e \t DevAcc: %.3f \t TestAcc: %.3f \t baseline: %.3f \n',n-nv, C_best , 1-err_best, 1-mean(YF<=0), max(mean(YTest >0), mean(YTest <0)));


    end
    
    subplot(size(windows,2),1, numW); hold on; box on; grid on;
    plot(realY,'--r');
    plot(predictions,'g');
    %title(['[',num2str(),',',num2str(size(X,1)-nt*(kEE-1)),']']);
    hold off;
    numW = numW +1;
    
    fprintf('Estimated Accuracy: %.3f \t Baseline: %.3f \n',1-err, max(baseline,1-baseline));
    accuracies = [accuracies; 1-err];

end


figure; box on; grid on;hold on;
plot(windows, accuracies,'r');
plot(windows, 0.6*ones(size(windows)),'--k');

legend('Accuracy','baseline')
title('Test Accuracy varying the X average window');
hold off;




% 
% 
% 
% 
% 
% 
% 
% 
% C= 23.357214690901213;              
% gamma=6.951927961775605;  
% mov_avg_window = 120;
% 
% j = 1;
% finalAcc = 0;
% baseline = 0;
% nk=7;
% 
% for finalRemove = 0:100:(nk-1)*100
%     %tutte le volte scarto un pezzo dell'ultima parte del dataset in modo
%     %da cambiare test set
%     
%     X = movmean(Xoriginal,[mov_avg_window 0],1);
%     X = X(1:end-finalRemove,:);
%     Y = Yoriginal(1:end-finalRemove,:);
%     [n,d] = size(X);
%     
%     %Last part for testing
%     nt = 100;
%     XTest = X(end-nt:end,:);
%     YTest = Y(end-nt:end,:);
%     X = X(1:end-nt,:);
%     Y = Y(1:end-nt,:);
% 
%     [n,d] = size(X);
% 
% 
% 
%     H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
%     [~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));
% 
% 
% 
%     YP = exp(-gamma*pdist2(XTest,X))*diag(Y)*alpha+b;
%     subplot(3,ceil(nk/3),j); hold on; box on; grid on;
%     plot(YTest,'--r')
%     plot(YP,'g')
%     plot(zeros(size(YP)),'k');
%     
%     xlabel('time');
%     ylabel('prediction');
%     legend('y(t) = p(t) -p(t-30)','prediction','0')
% 
% 
%     YP = sign(YP);
%     acc = sum(YTest == YP)/size(YTest, 1);
%     fprintf('Test Accuracy: %.3f Baseline: %.3f  \n', acc, mean(YTest>0));
%     finalAcc = finalAcc + acc/nk;
%     baseline = baseline + mean(YTest>0)/nk;
% 
%     j = j+1;
% end
% fprintf('CV Accuracy: %.3f Baseline: %.3f  \n', finalAcc, baseline);
% 
% 
% 
% %% KERNEL PLOT
% 
% C= 23.357214690901213;              
% gamma=6.951927961775605;  
% mov_avg_window = 120;
% X = movmean(Xoriginal,[mov_avg_window 0],1);
% 
% 
% H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
% % diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
% [~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));
% 
% ns = 30000;
% 
% randPoints = 2*rand(ns,2)-1;
% XS = 2*rand(ns,d)-1;
% YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
% 
% 
% varName = {'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'};
% for i = 1:5
%     for j = 1:5
% %         XS = zeros(ns,d);
% %         XS(:,i) = randPoints(:,1);
% %         XS(:,j) = randPoints(:,2);
% %         YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
%         figure; hold on; box on; grid on;
%         title('Learned model: blue -> predict positive: accuracy: 0.60')
%         plot(XS(YS>0,i),XS(YS>0,j),'.c')
%         plot(XS(YS<0,i),XS(YS<0,j),'.m')
%         xlabel(varName{i})
%         ylabel(varName{j})
%     end
% end


%% Error Estimation with cross validation
gamma = 1.438450e-03;
C = 2.636651e+02;

H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
err = 0;
    
nv = round(n/2);
nv = round(0.7*n);


UB = C*ones(n,1);
UB(nv:end) = 0;

[~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

YP = exp(-gamma*pdist2(X(nv:end,:),X))*diag(Y)*alpha+b;
YF = YP.*Y(nv:end);

figure; hold on; box on; grid on;
plot(Y(nv:end,:),'--r');
plot(YP,'g');
plot(YP.*Y(nv:end,:),':b');


fprintf('Accuracy: %.3f\n',1 - mean(YF<=0));

ns = 10000;

XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;


varName = {'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'};
for i = 3:3
    for j = 4:4
        figure; hold on; box on; grid on;
        title('Learned model: blue -> predict positive: accuracy: 0.60')
        plot(XS(YS>0,i),XS(YS>0,j),'.c')
        plot(XS(YS<0,i),XS(YS<0,j),'.m')
        xlabel(varName{i})
        ylabel(varName{j})
    end
end








%% DISCOVERY: X averaged plot

%
% LA FINESTRA MIGLIORE SEMBRA ESSERE 30
% SENZA AVERAGING IN PRATICA STIAMO DANDO SOLO NOISE
% GIA CON WIN = 10 LA SITUAZIONE MIGLIORA MOLTISSIMO
%
%


colors = {'-b','-g','-m','-k','-y','-p','-c'};
varName = {'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'};
i = 1;
step = 100;


% for mov_avg_window = [1,10, 30, 60, 90, 120, 750]
%     figure; hold on; box on; grid on;
%     plot(1:size(X,1), Y,'--r')
%     title(mov_avg_window);
%     for varNum = 1:5
%         X = movmean(Xoriginal,[mov_avg_window 0],1);
%         plot(1:size(X,1), X(:,varNum), colors{varNum});
%     end
%     legend('Y', 'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY');
% end  


i = 1;
windows = [1,10, 30, 60, 90, 120];
for mov_avg_window = windows
    X = movmean(Xoriginal,[mov_avg_window 0],1);
    subplot(3,2,i);box on; grid on;
    title(['X mov avg: ', num2str(windows(i))]);
    hold on;
    plot(X(Y==+1,3),X(Y==+1,4),'ob', 'MarkerFaceColor','b', 'MarkerSize',8)
    plot(X(Y==-1,3),X(Y==-1,4),'or','MarkerFaceColor','r', 'MarkerSize',8)
    xlabel('NEGATIVE')
    ylabel('POSITIVE')
    hold off;
    i = i+1;
end  
