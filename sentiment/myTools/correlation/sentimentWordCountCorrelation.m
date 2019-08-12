%% SenticNet sentiment
%

clear; clc; close all;

%% Read Y
filename = '/home/simone/Desktop/NLFF/DataSetIndexesLabeled/indexesAAPL.csv';
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
filename = '/home/simone/Desktop/NLFF/sentiment/myTools/wordCount+KSVM/SentimentNews/AAPL.csv';
delimiter = ',';
startRow = 3;
formatSpec = '%*s%*s%f%f%f%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
Xall = [dataArray{1:end-1}];
clearvars filename delimiter startRow formatSpec fileID dataArray ans;
%%
[n,d] = size(Xall);


%Normalizzo
for j = 1:d
    mi = min(Xall(:,j));
    ma = max(Xall(:,j));
    di = ma - mi;
    if (di < 1.e-6)
        Xall(:,j) = 0;
    else
        Xall(:,j) = (Xall(:,j)-mi)/di;
    end
end

%%

features = {'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'};

for featureNumber = 1:d
    figure; box on; grid on;
    X = Xall(:,featureNumber);
    me = median(X);
    l = 1;
    for k=[1 10 30 60]
       
        Z = ones(size(Y,1),1);
        for j=k+1:n
            Z(j)=Y(j)-Y(j-k);
        end
        
        negMean = [0 0];
        negMean(1) = mean(X(X<me));
        negMean(2) = mean(Z(X<me));
        posMean = [0 0];
        posMean(1) = mean(X(X>me));
        posMean(2) = mean(Z(X>me));


        subplot(2,2,l)
        hold on;
        plot(X(:,1),Z,'.b')
        plot(X(:,1),zeros(size(X,1),1),'-k')
        plot(negMean(1),negMean(2),'or','LineWidth',4)
        plot(posMean(1),posMean(2),'og','LineWidth',4)

        hold off;
        ylim([-5 5])
        xlabel(features(featureNumber));
        title(['dP/dt : ' num2str(k) ' window MovAvg'])
        l = l+1;
    end
end

%% Pearson correlation
[n,d] = size(Xall);


%Normalizzo
for j = 1:d
    mi = min(Xall(:,j));
    ma = max(Xall(:,j));
    di = ma - mi;
    if (di < 1.e-6)
        Xall(:,j) = 0;
    else
        Xall(:,j) = (Xall(:,j)-mi)/di;
    end
end



features = {'CONSTRAINING','LITIGIOUS','NEGATIVE','POSITIVE','UNCERTAINTY'};
%%

for k=[1 10 30 60]
    Yavg = movmean(Y,k);
    Z = -ones(size(Y,1),1);
    for j=2:n
        Z(j)=Yavg(j)-Yavg(j-1);
    end
    for featureNumber = 1:d
        X = Xall(:,featureNumber);
        c = corr(X,Z);
        fprintf('%d) %s Correlation coeff: %f \n',k, features{featureNumber}, c);
    end
end
