
%% MAKE DATASET
clear; clc; close all;

%tickers = {'AAPL','ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALGN', 'ALXN', 'AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};
tickers = {'AAPL','ADBE', 'ADI', 'ADSK', 'AKAM', 'ALGN','AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN',  'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};

ticAccs = [];
bases = [];
 

for tic =1:size(tickers,2)
    keepvars = {'tickers','tic','ticAccs', 'bases'};
    clearvars('-except', keepvars{:});
    fprintf('%s \n',tickers{tic});
    tickerSentiment = importSentiment(strcat('SentimentNews2/',tickers{tic},'.csv'));

    tickerIndexes = importIndexes(strcat('/home/simone/Desktop/NLFF/indexes/indexes',tickers{tic},'.csv'));
    
    initDate = max([tickerSentiment.initTime(1) tickerIndexes.date(1)]);
    finalDate = min([tickerSentiment.initTime(end) tickerIndexes.date(end)]);
    
    tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
    tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.date<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.date>= initDate, :);
        
    X = [tickerSentiment.CONSTRAINING  tickerSentiment.LITIGIOUS tickerSentiment.NEGATIVE tickerSentiment.POSITIVE tickerSentiment.UNCERTAINTY];
    %X =  [tickerSentiment.NEGATIVE tickerSentiment.POSITIVE];
    Y =  [tickerIndexes.close];       
    
       


    fprintf("Dataset created \n");






    % -- NORMALIZATION & MOMENTUM

    momentum = 30;
    [n,d] = size(X);

    Z = -ones(size(Y,1),1);

    for i=1:n-momentum
        Z(i)=Y(i+momentum)-Y(i);
    end
    Y = sign(Z);
    Y(Y==0) = -1;
    % Rimuovo prima parte senza momentum window

    X = X(1:end-momentum,:);
    Y = Y(1:end-momentum,:);
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


    %%Retain test set:

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

 %%SVM SEQUENTIAL Model Selection


    nk= 4; %numero cross validation
    err_best = +Inf;
    windows = [1 3 5 10 20 30 60 120];
    splitting = floor(linspace(1, n, 1.6*nk));
    splitting = splitting(end-nk:end);
    %fprintf('nv: %d \n',splitting(2)-splitting(1));

    for mov_avg_window = windows
        fprintf('.');
        X = movmean(Xoriginal,[mov_avg_window 0],1);
        Y = Yoriginal;
        H = diag(Y)*(X*X')*diag(Y);
        for C = logspace(-6,3,30)
            %splitto il dataset
            err = 0;
            for k=1:nk
                v = splitting(k):splitting(k+1);
                
                %Balancing of validation set
                inverted = 0;
                if(sum(Y(v) < 0) > sum(Y(v) > 0))
                    Y = -1*Y;
                    inverted = 1;
                end
                index = Y(v) > 0;
                count = sum(Y(v) < 0);
                rp = randperm(size(index,1));
                i = 1;
                while(count > 0)
                    if(index(rp(i)))
                        index(rp(i)) = false;
                        count = count -1;
                    end
                    i = i+1;
                end
                v(index) = [];
                if(inverted == 1)
                    Y = -1*Y;
                end
                
                %fprintf('nv: %d 1: %d -1: %d\n',size(v,2), sum(Y(v) > 0), sum(Y(v) < 0))

                UB = C*ones(n,1);
                % mi baster� settare questi indici a 0 per non contarare i punti nel training
                UB(v(1):end) = 0;

                %Balancing of training set
                inverted = 0;
                if(sum(Y(1:v(1)) < 0) > sum(Y(1:v(1)) > 0))
                    Y = -1*Y;
                    inverted = 1;
                end
                index = Y(1:v(1)) > 0;
                count = sum(Y(1:v(1)) < 0);
                rp = randperm(size(index,1));
                i = 1;
                while(count > 0)
                    if(index(rp(i)))
                        index(rp(i)) = false;
                        count = count -1;
                    end
                    i = i+1;
                end
                UB(index) = 0;
                if(inverted == 1)
                    Y = -1*Y;
                end
                % fprintf('nt: %d\n',sum(UB>0))

                [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

                YP = X(v,:)*X'*diag(Y)*alpha+b;
                YF = YP.*Y(v);
                %YF = H*alpha+b; %classifico tutti i punti
                err = err+mean(YF<=0)/nk;
            end
            %fprintf('%d %e %.3f %.3f\n',mov_avg_window, C, err, err_best);
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
    fprintf('\nAccuracy: %.3f Baseline: %.3f (C = %.3f Window = %d) \n ',mean(YF>0), mean(YTest>0), C_best, win_best);
    ticAccs = [ticAccs; mean(YF>0)];
    figure; hold on; box on; grid on;
    plot(YTest,'--r');
    plot(YP,'g');
    plot(zeros(size(YP)),'--k')
    title([' Test predictions, Window: ',num2str(win_best) ,' Accuracy: ' num2str(floor(mean(YF>0)*100)),'% baseline: 62%'])
    legend('sign(p(t)-p(t-30))', 'prediction', '0')
    confusionmat(YTest, sign(YP),'Order',[-1 1])


    % valori di default sono 30% test se voglio fare model selection
    % 20% test 20% validation se voglio fare error estimation

end
fprintf('Avg acc: %.3f', ticAccs/size(ticAccs,1));
