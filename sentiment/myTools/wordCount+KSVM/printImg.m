clear; clc; close all;

shift=0;
tic = 1;
tickers = {'AAPL'};

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
for mov_avg_window = [30]
    %subplot(2,2,i); hold on;
    figure; hold on; box on; grid on;
    plot(timeSteps(momentum+1000:end), Y(1001:end), 'm');
    plot(timeSteps(momentum+1000:end),Z(1001:end), 'b');
    Xavg = movmean(X,[mov_avg_window 0],1);
    plot(timeSteps(momentum+1000:end),2*(Xavg(momentum+1000:end,3)+0.5),'--r');
    plot(timeSteps(momentum+1000:end),(Xavg(momentum+1000:end,4)+0.5)*2,'--g');
    set(gca, 'YTick', []);
    ylim([-1.1, 1.1])
    %title(mov_avg_window)
    hold off;
    lgd = legend('p(t) - p(t-30)','y','NEGATIVE','POSITIVE', 'Location','northwest');
    lgd = legend('p(t) - p(t-30)','y', 'Location','northwest');
    lgd.FontSize = 14;
    i = i+1;
end
