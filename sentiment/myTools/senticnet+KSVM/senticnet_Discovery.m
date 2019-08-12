%% MAKE DATASET
clear; clc; close all;


tickers = {'AAPL','ADBE', 'ADI', 'ADSK', 'AKAM', 'ALGN','AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN',  'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};
%tickers = {'ADP','ALXN' , 'CHTR''CMCSA', 'REGN',}; questi non funzionano
ticAccs = [];
bases = [];
tickers = {'AAPL'};


for tic =1:size(tickers,2)
    keepvars = {'tickers','tic','ticAccs', 'bases'};
    clearvars('-except', keepvars{:});
    fprintf('%s \n',tickers{tic});
    tickerSentiment = importSentiment(strcat(tickers{tic},'.csv'));

%     
%     check possible error!!!!!!!!!
%     are the sent and index on the same scale?
%     
%     
%     
%    
    
    
    
    tickerIndexes = importIndexes(strcat('/home/simone/Desktop/NLFF/DataSetIndexes/indexes',tickers{tic},'.csv'));
    initDate = max([tickerSentiment.initTime(1) tickerIndexes.data(1)]);
    finalDate = min([tickerSentiment.initTime(end) tickerIndexes.data(end)]);

    tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
    tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data<= finalDate, :);
    tickerIndexes = tickerIndexes(tickerIndexes.data>= initDate, :);

    X = [tickerSentiment.NEGATIVE tickerSentiment.POSITIVE];
    Y = [tickerIndexes.close];       








    fprintf("Dataset created \n");



    momentum = 30;

    [n,d] = size(X);

    Z = -ones(size(Y,1),1);

    for i=momentum+1:n
        Z(i)=Y(i)-Y(i-momentum);
    end

    Y = 2*(Y(momentum:end,:)-min(Y(momentum:end,:)))/(max(Y(momentum:end,:))-min(Y(momentum:end,:)))-1;
    Z = sign(Z(momentum:end,:));

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


    i =1;
    for mov_avg_window = [1 10 30 60]
        subplot(2,2,i);
        %figure;
         hold on;
        plot(Y, 'm');
        plot(Z, 'b');
        Xavg = movmean(X,[mov_avg_window 0],1);
        plot(Xavg(momentum:end,1),'--r');
        plot(Xavg(momentum:end,2)+0.15,'--g');
        title(mov_avg_window)
        hold off;
        legend('p(t) - p(t-30)','y','#NEG','#POS');
        i = i+1;
    end
    


    % % Predictor: market up if #positive > #neg

    Y =Z;

    mov_avg_window = 30;
    fprintf('Baseline: %.3f\n',mean(Y>0));
    bases = [bases; mean(Y>0)]; 
    %Non c'e training percio tutti i punti sono per test

    accs = [];
    biases = [];
    windows = [1 10 20 30 40 50 60 80 90 120];
    for mov_avg_window = windows
        Xavg = movmean(X,[mov_avg_window 0],1);
        Xavg = Xavg(momentum:end,:);
        v = floor(size(Xavg,1)*0.3);

        best_acc = -Inf;
        %'Validation'
        for bias = linspace(0,0.3,1000)
            YP = [];

            for i=1:v
                if(Xavg(i,2)+bias>Xavg(i,1))
                    YP = [YP; 1];
                else
                    YP = [YP; -1];
                end
            end

            YF = YP.*Y(1:v);
            acc = mean(YF>0);
            if(best_acc < acc)
                best_acc = acc;
                best_bias = bias;
            end
        end

        %test
        YP = [];
        for i=v+1:size(Xavg,1)
            if(Xavg(i,2)+best_bias>Xavg(i,1))
                YP = [YP; 1];
            else
                YP = [YP; -1];
            end
        end

        YF = YP.*Y(v+1:end);
        acc = mean(YF>0);
        accs = [accs; acc];
        biases = [biases, best_bias];

    end
    [maxAcc, index] = max(accs);
    ticAccs = [ticAccs; maxAcc];
    fprintf('Accuacy: %.3f Window avg: %d bias: %f\n',maxAcc, windows(index), biases(index));
    % figure; grid on;
    % plot(windows,accs);

    title('Test accuracy varying x window average')
    Xavg = movmean(X,[mov_avg_window 0],1);
    Xavg = Xavg(momentum:end,:);

end
figure; grid on; box on; hold on;
plot(ticAccs,'r');
plot(bases,'--k');
title('Accuracies for different stock')
legend('accuracy', 'base');
