%% SenticNet sentiment
%

clear; clc; close all;

%% Read Y
filename = '/home/simone/Scrivania/NLFF/DataSetIndexes/indexesAAPL.csv';
delimiter = ',';
startRow = 998;
endRow = 1735;
formatSpec = '%*s%f%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%*s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, endRow-startRow+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
Y = [dataArray{1:end-1}];
clearvars filename delimiter startRow endRow formatSpec fileID dataArray ans;

%% Read X
filename = '/home/simone/Scrivania/NLFF/sentiment/myTools/correlation/sentimentSenticNetAAPL.csv';
delimiter = ',';
startRow = 2;
formatSpec = '%*s%*s%f%*s%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
X = [dataArray{1:end-1}];
clearvars filename delimiter startRow formatSpec fileID dataArray ans; 
%%

[n,d] = size(X);

%Normalizzo
for j = 1:d
    mi = min(X);
    ma = max(X);
    di = ma - mi;
    if (di < 1.e-6)
        X = 0;
    else
        X = (X - mi)/di;
    end
end
me = median(X);

means = [];

%Y = Z;
figure; box on; grid on;
i = 1;
for k=[1 10 30 60 90 120]
    Yavg = movmean(Y,k);
    Z = -ones(size(Y,1),1);
    for j=2:n
        Z(j)=Yavg(j)-Yavg(j-1);
    end
    negMean = [0 0];
    negMean(1) = mean(X(X<me));
    negMean(2) = mean(Z(X<me));
    posMean = [0 0];
    posMean(1) = mean(X(X>me));
    posMean(2) = mean(Z(X>me));
    means = [means; posMean(2) negMean(2)];
    
    subplot(3,2,i)
    hold on;
    plot(X(:,1),Z,'.b')
    plot(X(:,1),zeros(size(X,1),1),'-k')
    plot(negMean(1),negMean(2),'or','LineWidth',4)
    plot(posMean(1),posMean(2),'og','LineWidth',4)
    
    hold off;
    ylim([-1 1])
    xlabel('SenticNet Sentiment');
    title(['dP/dt with: ' num2str(k) ' window Moving Average'])
    i = i+1;
end
fprintf("Normalization Done \n")
