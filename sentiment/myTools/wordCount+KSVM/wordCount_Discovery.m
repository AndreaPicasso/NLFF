%% MAKE DATASET
clear; clc; close all;


tickers = {'AAPL','ADBE', 'ADI', 'ADSK', 'AKAM', 'ALGN','AMAT', 'AMGN', 'AMZN', 'ATVI', 'AVGO', 'BIDU',  'BIIB', 'BMRN', 'CA', 'CELG', 'CERN',  'COST', 'CSCO', 'CSX', 'CTAS', 'CTRP', 'CTSH', 'CTXS','DISCA', 'DISH', 'DLTR', 'EA', 'EBAY', 'ESRX', 'EXPE','FAST', 'FB', 'FISV', 'FOXA', 'GILD', 'GOOGL', 'HAS','HOLX', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU','ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LRCX', 'MAR','MAT', 'MCHP', 'MDLZ', 'MELI', 'MNST', 'MSFT', 'MU', 'MXIM','MYL', 'NFLX', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PCLN', 'PYPL', 'QCOM', 'ROST', 'SBUX', 'SHPG', 'SIRI', 'STX','SWKS', 'SYMC', 'TMUS', 'TSCO', 'TSLA', 'TXN', 'ULTA', 'VIAB', 'VOD', 'VRSK', 'VRTX', 'WBA', 'WYNN', 'XLNX', 'XRAY'};
tickers = {'AAPL','AMZN','GOOGL','MSFT','FB','INTC','CSCO','CMCSA','NVDA','NFLX'};
tickers = {'AAPL'};
shifting = [0, 7, 14, 21, 35, 70, 105,210];
shifting = [0];
%figure; grid on; box on; hold on;
for tic =1:size(tickers,2)
    ticAccs = [];
    bases = [];
    for shift = shifting
        %keepvars = {'tickers','tic','ticAccs', 'bases','shifting','shift'};
        %clearvars('-except', keepvars{:});
        fprintf('%s \n',tickers{tic});
        tickerSentiment = importSentiment(strcat('SentimentSingleNewsFullNoNorm/',tickers{tic},'.csv'));

        tickerIndexes = importIndexes(strcat('/home/simone/Desktop/NLFF/indexes/indexes',tickers{tic},'.csv'));
        tickerSentiment.initTime = tickerSentiment.initTime + hours(shift);
        initDate = max([tickerSentiment.initTime(1) tickerIndexes.date(1)]);
        finalDate = min([tickerSentiment.initTime(end) tickerIndexes.date(end)]);

        %tickerSentiment = tickerSentiment(tickerSentiment.initTime>= initDate, :);
        %tickerSentiment = tickerSentiment(tickerSentiment.initTime<= finalDate, :);
        %tickerIndexes = tickerIndexes(tickerIndexes.date<= finalDate, :);
        %tickerIndexes = tickerIndexes(tickerIndexes.date>= initDate, :);

        X = [];
        Y = [];
        timeSteps = [];
        i = 1;
        j = 1;
        while(tickerIndexes.date(j) < initDate)
                j=j+1;
        end
        while(tickerSentiment.initTime(i) < initDate)
                i=i+1;
        end

        while(tickerIndexes.date(j) < finalDate && tickerSentiment.initTime(i) < finalDate)
            timeSlotX = [];
            while(i < size(tickerSentiment,1) &&  tickerIndexes.date(j) > tickerSentiment.initTime(i))
                timeSlotX = [timeSlotX; [tickerSentiment.CONSTRAINING(i)  tickerSentiment.LITIGIOUS(i) tickerSentiment.NEGATIVE(i) tickerSentiment.POSITIVE(i) tickerSentiment.UNCERTAINTY(i)]];
                i = i+1;
            end
            if(size(timeSlotX,1)>1)
                X = [X; mean(timeSlotX)];
            else if(size(timeSlotX,1)==1)
                 X = [X; timeSlotX];
                else
                  X = [X; [0 0 0 0 0]];  
                end
            end
             Y = [Y; tickerIndexes.close(j)];
             timeSteps = [timeSteps; tickerIndexes.date(j)];
             j=j+1;
        end


        %X = [tickerSentiment.CONSTRAINING  tickerSentiment.LITIGIOUS tickerSentiment.NEGATIVE tickerSentiment.POSITIVE tickerSentiment.UNCERTAINTY];
        %X = [X; [tickerSentiment.NEGATIVE tickerSentiment.POSITIVE]];
        %Y = [tickerIndexes.close];       




        fprintf("Dataset created \n");



        momentum = 30;

        [n,d] = size(X);

        Z = -ones(size(Y,1),1);

        for i=momentum+1:size(Y,1)
            Z(i)=Y(i)-Y(i-momentum);
        end

        Y = 2*(Y(momentum:end,:)-min(Y(momentum:end,:)))/(max(Y(momentum:end,:))-min(Y(momentum:end,:)))-1;
        Z = sign(Z(momentum:end,:));
        plot(Y)
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
        figure;
        for mov_avg_window = [1 10 20 30]
            %subplot(2,2,i); hold on;
            figure; hold on; box on; grid on;
            plot(timeSteps(momentum:end), Y, 'm');
            plot(Z, 'b');
            Xavg = movmean(X,[mov_avg_window 0],1);
            plot(Xavg(momentum:end,3),'--r');
            plot(Xavg(momentum:end,4),'--g');
            %title(mov_avg_window)
            hold off;
            legend('p(t) - p(t-30)','y','NEGATIVE','POSITIVE');
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
                    if(Xavg(i,4)+bias>Xavg(i,3))
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
                if(Xavg(i,4)+best_bias>Xavg(i,3))
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

        % title('Test accuracy varying x window average')
        Xavg = movmean(X,[mov_avg_window 0],1);
        Xavg = Xavg(momentum:end,:);
    end
%     col = rand(1,3);
%     plot(ticAccs,'color',col,'LineWidth',2);
%     plot(bases,'--','color',col);
    %title('Accuracies for different stock')
    

end
legend('AAPL','','AMZN','','GOOGL','','MSFT','','FB','','INTC','','CSCO','','CMCSA','','NVDA','','NFLX','');
