
%% MAKE DATASET
clear; clc; close all;

load('dataset.mat');
fprintf("Dataset created \n");
      

%% NORMALIZATION & MOMENTUM
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


fprintf("Normalization Done \n")


%% Make classes balanced:
% if(sum(Y<0) < sum(Y>0))
%     x = X(Y<0,:);
%     y = Y(Y<0,:);
% 
%     index = Y>0;
%     count = size(x,1);
%     for i=1:size(index,1)
%         if(count == 0)
%             index(i) = 0;
%         end
%         if(index(i) >0)
%             count = count -1;
%         end
%     end
%     X = [x; X(index,:)];
%     Y = [y; Y(index,:)];
% else
%     x = X(Y>0,:);
%     y = Y(Y>0,:);
% 
%     index = Y<0;
%     count = size(x,1);
%     for i=1:size(index,1)
%         if(count == 0)
%             index(i) = 0;
%         end
%         if(index(i) >0)
%             count = count -1;
%         end
%     end
%     X = [x; X(index,:)];
%     Y = [y; Y(index,:)];
% end
% [n,d] = size(X);
% 
% Xoriginal = X;
% Yoriginal = Y;

%% Retain test
i = randperm(n);
nt = round(.2*n);
XTest = X(i(1:nt),:);
YTest = Y(i(1:nt),:);
X = X(i(nt:end),:);
Y = Y(i(nt:end),:);

[n,d] = size(X);

%%
nk= 8; %numero cross validation

gammaSpace = logspace(-6,4,10);
CSpace = logspace(-9,4,50);

err_best = +Inf;
train_errs = zeros(size(gammaSpace,2),size(CSpace,2));
test_errs = zeros(size(gammaSpace,2),size(CSpace,2));

splitting = floor(linspace(1, n, 1.6*nk));
splitting = splitting(end-nk:end);
fprintf('nv: %d \n',splitting(2)-splitting(1));

Q = pdist2(X,X);
for g = 1:size(gammaSpace,2)
    gamma = gammaSpace(g);
    H = diag(Y)*exp(-gamma*Q)*diag(Y);
    for c = 1:size(CSpace,2)
        C = CSpace(c);
        %splitto il dataset
        err = 0;
        tr_err = 0;
        baseline = 0;
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


            [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

            %Training Error
            YP = exp(-gamma*pdist2(X(UB~= 0 ,:),X))*diag(Y)*alpha+b;
            YF = YP.*Y(UB~= 0);
            tr_err = tr_err+mean(YF<=0)/nk;
            
            YP = exp(-gamma*pdist2(X(v,:),X))*diag(Y)*alpha+b;
            YF = YP.*Y(v);
            %YF = H*alpha+b; %classifico tutti i punti
            err = err+mean(YF<=0)/nk;
            baseline = baseline +mean(Y(v)<=0)/nk; 
        end
        test_errs(g,c) = err;
        train_errs(g,c) = tr_err;
        fprintf('C: %e g:%e err: %.3f err*: %.3f base: %.3f\n', C,gamma, err, err_best, baseline);
        if(err_best > err)
            gamma_best = gamma;
            err_best = err;
            C_best = C;
        end
    end
end
% valori di default sono 30% test se voglio fare model selection
% 20% test 20% validation se voglio fare error estimation
fprintf('OPT: C = %e  gamma = %e\n', C_best,gamma_best );
% semilogx(logspace(-9,3,50), 1-train_errs, 'b', logspace(-9,3,50), 1-test_errs, 'r',logspace(-9,3,50), 0.759*ones(size(test_errs)), 'k--');
% title('SVM')
% legend('train accuracy', 'test accuracy', 'base');
% xlabel('C');

%%
semilogx(CSpace, 1-min(train_errs), 'b', CSpace, 1-min(test_errs), 'r',CSpace, 0.759*ones(size(test_errs, 2)), 'k--')
ylim([0,1]);
title('KSVM varying C with optimal gamma')
legend('train accuracy', 'test accuracy', 'base');
xlabel('C');


semilogx(gammaSpace, 1-min(train_errs,[],2), 'b', gammaSpace, 1-min(test_errs,[],2), 'r',gammaSpace, 0.759*ones(size(test_errs,1),1), 'k--')
ylim([0,1]);
title('KSVM varying gamma with optimal C')
legend('train accuracy', 'test accuracy', 'base');
xlabel('gamma');
%% Model Selection 

nk= 30; %numero cross validation
err_best = +Inf;



Q = pdist2(X,X);
for g = 1:size(gammaSpace,1)
    gamma = gammaSpace(g);
    H = diag(Y)*exp(-gamma*Q)*diag(Y);
    for c = 1:size(CSpace,1)
        
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


%% TEST ERROR
%a questo punto posso rifare l'ottimizzazione con i parapetri trovati
C=C_best;
gamma=gamma_best;
%H = diag(Y)*(X*X')*diag(Y); sostinuiamo con il kernel
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);ns = 100000;
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));



YP = exp(-gamma*pdist2(XTest,X))*diag(Y)*alpha+b;
YP = sign(YP);
acc = sum(YTest == YP)/size(YTest, 1);
fprintf('Test Accuracy: %.3f \n', acc);



%% Error Estimation (with model selection)
% 



predictions = [];
realY = [];
permT = [];

nt = floor(.2*n);
nkEE = floor(n/nt);
fprintf('fold: %d \n',nkEE);
permTest = randperm(n);

err = 0;
baseline = 0;
for kEE=1:nkEE
    
    %Retain Test part
    XTest = Xoriginal(permTest(1+(kEE-1)*nt:(kEE)*nt),:);
    YTest = Yoriginal(permTest(1+(kEE-1)*nt:(kEE)*nt),:);
    
    
    X = Xoriginal;
    X(permTest(1+(kEE-1)*nt:(kEE)*nt),:) = [];
    Y = Yoriginal;
    Y(permTest(1+(kEE-1)*nt:(kEE)*nt),:) = [];
    [n,d] = size(X);
    
    % Model Selection:
    nk= 30; %numero cross validation
    err_best = +Inf;
    Q = pdist2(X,X);
    for gamma = logspace(-6,4,20)
        fprintf('.');
        H = diag(Y)*exp(-gamma*Q)*diag(Y);
        for C = logspace(-6,4,20)
            %splitto il dataset
            errVal = 0;
            for k=1:nk
                i = randperm(n);
                nv = round(.2*n); %30% dei dati per la validation
                UB = C*ones(n,1);
                % mi baster� settare questi indici a 0 per non contarare i punti nel training
                UB(i(1:nv)) = 0;

                [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));

                YP = exp(-gamma*pdist2(X(i(1:nv),:),X))*diag(Y)*alpha+b;
                YF = YP.*Y(i(1:nv));
                %YF = H*alpha+b; %classifico tutti i punti
                errVal = errVal+mean(YF<=0)/nk; %considero solo quelli di validation

            end
            %fprintf('%e %e %.3f %.3f\n', gamma, C, errVal, err_best);
            if(err_best > errVal)
                %se miglioro l'errore best aggiorno
                err_best = errVal;
                C_best = C;
                gamma_best = gamma;
            end
        end
    end     
    
    UB = C_best*ones(n,1);
    H = diag(Y)*exp(-gamma_best*Q)*diag(Y);
    
    [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
    YP = exp(-gamma_best*pdist2(XTest,X))*diag(Y)*alpha+b;
    
    predictions = [predictions; YP ];
    permT = [permT permTest(1+(kEE-1)*nt:(kEE)*nt)];
    realY = [realY; YTest];
    
    YF = YP.*YTest;
    err = err+mean(YF<=0)/nkEE; %considero solo quelli di validation
    baseline = baseline + mean(YTest >0)/nkEE;
    
    fprintf('\n nTrain: %d \t (C = %e, gamma = %e) \t DevAcc: %.3f \t TestAcc: %.3f \t baseline: %.3f \n',n-nv, C_best ,gamma_best, 1-err_best, 1-mean(YF<=0), max(mean(YTest >0), mean(YTest <0)));

    
end

%reverse perm
pt = -ones(size(permT));
for i = 1:size(permT,1)
    pt(permT(i)) = i;
end
figure; hold on; box on; grid on;
title('Predictions')
plot(realY(pt),'--r');
plot(predictions(pt),'g');
hold off;

fprintf('Estimated Accuracy: %.3f \t Baseline: %.3f \n',1-err, max(baseline, 1-baseline));






