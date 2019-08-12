clear; clc; close all;

tickers = {'AAPL','ADBE', 'ADI', 'ADP', 'ADSK', 'AKAM', 'ALGN', 'ALXN', 'AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};
 
for tick =1:size(tickers,2)
    X = [];
    Y = [];
    
    fprintf('-------- %s dataset created \n',tickers{tick});
    tickerSentiment = importSentiment(strcat('SentimentNews/',tickers{tick},'.csv'));

    tickerIndexes = importIndexes(strcat('/home/simone/Desktop/NLFF/DataSetIndexesLabeled/indexes',tickers{tick},'.csv'));
    initDate = max([tickerSentiment.initTime(1) tickerIndexes.data(1)]);
    finalDate = min([tickerSentiment.initTime(end) tickerIndexes.data(end)]);
    
    tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
    tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data>= initDate, :);
        
    X = [X; [tickerSentiment.CONSTRAINING  tickerSentiment.LITIGIOUS tickerSentiment.NEGATIVE tickerSentiment.POSITIVE tickerSentiment.UNCERTAINTY]];
    Y = [Y; tickerIndexes.close];       
    
    
    momentum = 30;
    mov_avg_window =30;
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
    %----------MOVING AVERAGE
    X = movmean(X,[mov_avg_window 0],1);
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
    nt = round(.3*n);
    i = randperm(n);
    XTest = X(i(1:nt),:);
    YTest = Y(i(1:nt),:);
    X = X(i(nt:end),:);
    Y = Y(i(nt:end),:);
    [n,d] = size(X);
    
    nk= 40; %numero cross validation
    err_best = +Inf;

    Q = pdist2(X,X);
    for gamma = logspace(-6,4,20)
       fprintf(".")
        H = diag(Y)*exp(-gamma*Q)*diag(Y);
        for C = logspace(-6,4,20)
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
            % fprintf('%e %e %.3f %.3f\n', gamma, C, err, err_best);
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
    fprintf(' OPT: C = %e gamma = %e DEV accuracy = %.3f\n', C_best, gamma_best, 1-err_best);
    
    % --- TEST
    C=C_best;                       
    gamma=gamma_best;               
    H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
    [~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));

    YP = exp(-gamma*pdist2(XTest,X))*diag(Y)*alpha+b;
    YP = sign(YP);
    acc = sum(YTest == YP)/size(YTest, 1);
    fprintf('Test Accuracy: %.3f \n', acc);

       
end

% -------- AAPL dataset created 
% .................... OPT: C = 7.847600e+01 gamma = 1.832981e-01 DEV accuracy = 0.935
% Test Accuracy: 0.953 
% -------- ADBE dataset created 
% .................... OPT: C = 2.636651e+02 gamma = 6.158482e-01 DEV accuracy = 0.909
% Test Accuracy: 0.911 
% -------- ADI dataset created 
% .................... OPT: C = 1.000000e+04 gamma = 1.832981e-01 DEV accuracy = 0.910
% Test Accuracy: 0.913 
% -------- ADP dataset created 
% .................... OPT: C = 2.976351e+03 gamma = 2.069138e+00 DEV accuracy = 0.899
% Test Accuracy: 0.929 
% -------- ADSK dataset created 
% .................... OPT: C = 2.976351e+03 gamma = 1.623777e-02 DEV accuracy = 0.906
% Test Accuracy: 0.939 
% -------- AKAM dataset created 
% .................... OPT: C = 1.000000e+04 gamma = 5.455595e-02 DEV accuracy = 0.915
% Test Accuracy: 0.917 
