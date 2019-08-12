%% Model Selection -> cerco il miglior C, gamma

%abbiamo due strade migliori: bootstrap e cross validation, usiamo cross
%validation
nk= 10; %numero cross validation
%Nella realtà 10 è molto poco, almeno ci vuole un 100
err_best = +Inf;

%Notiamo che non dobbiamo calcolare le distanze tutte le volte, mi basterà
%farlo una volta all'inizio
Q = pdist2(X,X);
% /!\ Perchè sono partito da soluzioni piccole andando verso soluzioni
% grandi? perchè applico il rasoio di Okkam, cioè a parità di errore scelgo
% sempre la soluzione più semplice (aggiorno il best solo se err > err_best)
% e non >=, err_best è stato ottenuto con parametri più piccoli

%Il fatto che sia semplice o vedo perchè che il margine è
%abbastanza grande cioè predico blu esattamente solo quando ci sono tanti
%punti blu (idem per rosso) se non cerco di tenere il margine grande. Se
%invertissi i logspace avrei la soluzione + complessa (a minor errore) x cui
%avrei una figura a minor margine e piu strana
a =logspace(5,8,30);
range = combvec(a,a);
[~, I] = sort(sum(range),'descend');
for r = I
    gamma = range(2,r);
    C = range(1,r);
    H = diag(Y)*exp(-gamma*Q)*diag(Y);
    %splitto il dataset
    err = 0;
    for k=1:nk
        i = randperm(n);
        nv = round(.3*n); %30% dei dati per la validation
        UB = C*ones(n,1);
        % mi basterà settare questi indici a 0 per non contarare i punti nel training
        UB(i(1:nv)) = 0;
        %SMO_ab non funziona perchè abbiamo inserito dei vincoli tipo
        % 0 <= alpha_i <= 0 che devono essere gestiti, deve essere
        % settato alpha_i = 0
        [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
        % Ypred = exp(-gamma*Q)*diag(Y)*alpha;
        %devo contare errore se Y.*Ypred < 0 ma Y.*Ypred*alpha =
        %H*alpha
        YF = H*alpha+b; %classifico tutti i punti
        err = err+mean(YF(i(1:nv))<=0)/nk; %considero solo quelli di validation
    end
    fprintf('%e %e %.3f %.3f\n', gamma, C, err, err_best);
    if(err_best > err)
        %se miglioro l'errore best aggiorno
        err_best = err;
        C_best = C;
        gamma_best = gamma;
        end
end     
% valori di default sono 30% test se voglio fare model selection
% 20% test 20% validation se voglio fare error estimation
fprintf('OPT: C = %e gamma = %e', C_best, gamma_best);

%a questo punto posso rifare l'ottimizzazione con i parapetri trovati
C=C_best;
gamma=gamma_best;
%H = diag(Y)*(X*X')*diag(Y); sostinuiamo con il kernel
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
% diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));
%----- plot ----

% Notiamo che la soluzione è una soluzione di sicurezza, cioè ha il margine
% molto largo -> predico blu solo esattamente quando ci sono tanti punti
% blu (idem per rosso), se no tento di tenere il margine più largo possibile
% questo è perchè abbiamo scelto la soluzione a minimo C, gamma -> più
% semplice possibile

%Se vediamo lo spazio bi dimensionale [C, gamma] non ci sarà una soluzione
%ottima, perchè entrambi possono essere usati più o meno per regolarizzare, vi sarà
%una linea ottima in tale spazio
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;

figure, hold on, box on, grid on
plot(XS(YS>0,1),XS(YS>0,2),'.c')
plot(XS(YS<0,1),XS(YS<0,2),'.m')
plot(XS(YS==0,1),XS(YS==0,2),'.g')
plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
plot(XS(YS<-1,1),XS(YS<-1,2),'.r')

% Nota: alcune volte la soluzione non viene corretta, è sensato ciò?
 
plot(X(Y==+1,1),X(Y==+1,2),'ob','markersize',8,'linewidth',8)
plot(X(Y==-1,1),X(Y==-1,2),'or','markersize',8,'linewidth',8)
 
plot(X(alpha==C,1),X(alpha==C,2),'*g','markersize',8,'linewidth',8)
plot(X(alpha==0,1),X(alpha==0,2),'*y','markersize',8,'linewidth',8)
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k','markersize',8,'linewidth',8)



%% Model Selection -> cerco il miglior C, gamma

clear; clc; close all;

n=200;
d=2;
a = 0;
b= 1;
theta = linspace(0,2*2*pi,n/2)'; %angolo da girare: faccio 2 giri completi di 2pi
r = a+b*theta;
X = [r.*cos(+theta), r.*sin(+theta);...
    -r.*cos(-theta), r.*sin(-theta)];
Y = [-ones(n/2,1);ones(n/2,1)];

%Normalizzo
for j = 1:d
    mi = min(X(:,j));
    ma = max(X(:,j));
    di = ma - mi;
    if (di < 1.e-6)
        X(:,j) = 0;
    else
        X(:,j) = 2*(X(:,j)-mi)/di-1;
    end
end

%%

%abbiamo due strade migliori: bootstrap e cross validation, usiamo cross
%validation
nk= 10; %numero cross validation
%Nella realtà 10 è molto poco, almeno ci vuole un 100
err_best = +Inf;

%Notiamo che non dobbiamo calcolare le distanze tutte le volte, mi basterà
%farlo una volta all'inizio
Q = pdist2(X,X);
% /!\ Perchè sono partito da soluzioni piccole andando verso soluzioni
% grandi? perchè applico il rasoio di Okkam, cioè a parità di errore scelgo
% sempre la soluzione più semplice (aggiorno il best solo se err > err_best)
% e non >=, err_best è stato ottenuto con parametri più piccoli

%Il fatto che sia semplice o vedo perchè che il margine è
%abbastanza grande cioè predico blu esattamente solo quando ci sono tanti
%punti blu (idem per rosso) se non cerco di tenere il margine grande. Se
%invertissi i logspace avrei la soluzione + complessa (a minor errore) x cui
%avrei una figura a minor margine e piu strana

C = 6.7e+00;
gamma = 1.6e-01;

%for gamma = logspace(-6,5,30)%logspace(-6,4,30)
    for C = logspace(-6,5,30)%logspace(-6,4,30)
        H = diag(Y)*exp(-gamma*Q)*diag(Y);
        %splitto il dataset
        err = 0;
        for k=1:nk
            i = randperm(n);
            nv = round(.3*n); %30% dei dati per la validation
            UB = C*ones(n,1);
            % mi basterà settare questi indici a 0 per non contarare i punti nel training
            UB(i(1:nv)) = 0;
            %SMO_ab non funziona perchè abbiamo inserito dei vincoli tipo
            % 0 <= alpha_i <= 0 che devono essere gestiti, deve essere
            % settato alpha_i = 0
            [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y,UB,1e+8,1e-3,zeros(n,1));
            % Ypred = exp(-gamma*Q)*diag(Y)*alpha;
            %devo contare errore se Y.*Ypred < 0 ma Y.*Ypred*alpha =
            %H*alpha
            YF = H*alpha+b; %classifico tutti i punti
            err = err+mean(YF(i(1:nv))<=0)/nk; %considero solo quelli di validation
        end
        fprintf('%e %e %.3f %.3f\n', gamma, C, err, err_best);
        if(err_best > err)
            %se miglioro l'errore best aggiorno
            err_best = err;
            C_best = C;
            gamma_best = gamma;
        end
    end
%end     
% valori di default sono 30% test se voglio fare model selection
% 20% test 20% validation se voglio fare error estimation
fprintf('OPT: C = %e gamma = %e', C_best, gamma_best);

%a questo punto posso rifare l'ottimizzazione con i parapetri trovati
C=C_best;
gamma=gamma_best;
%H = diag(Y)*(X*X')*diag(Y); sostinuiamo con il kernel
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
% diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));
%----- plot ----

% Notiamo che la soluzione è una soluzione di sicurezza, cioè ha il margine
% molto largo -> predico blu solo esattamente quando ci sono tanti punti
% blu (idem per rosso), se no tento di tenere il margine più largo possibile
% questo è perchè abbiamo scelto la soluzione a minimo C, gamma -> più
% semplice possibile

%Se vediamo lo spazio bi dimensionale [C, gamma] non ci sarà una soluzione
%ottima, perchè entrambi possono essere usati più o meno per regolarizzare, vi sarà
%una linea ottima in tale spazio
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;

figure, hold on, box on, grid on
plot(XS(YS>0,1),XS(YS>0,2),'.c')
plot(XS(YS<0,1),XS(YS<0,2),'.m')
plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
plot(XS(YS<-1,1),XS(YS<-1,2),'.r')

% Nota: alcune volte la soluzione non viene corretta, è sensato ciò?
 
plot(X(Y==+1,1),X(Y==+1,2),'ob','markersize',8,'linewidth',8)
plot(X(Y==-1,1),X(Y==-1,2),'or','markersize',8,'linewidth',8)
 
plot(X(alpha==C,1),X(alpha==C,2),'*g','markersize',8,'linewidth',8)
plot(X(alpha==0,1),X(alpha==0,2),'*y','markersize',8,'linewidth',8)
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k','markersize',8,'linewidth',8)
