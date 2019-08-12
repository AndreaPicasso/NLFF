
%% SVM lineare biclasse
clear; clc; close all;

n=100;
d=2;
s=2; %distanza tra le classi

X = [randn(n/2,d)+s; randn(n/2,d)-s];
Y = [-ones(n/2,1);ones(n/2,1)]; %SVM vuole per forza le classi classificate come {-1,1}
% SVM risolve un problema di ottimizzazione quadratica, con la matrice Q (hessiano) se questa matrice � fatta male ci
% mette di pi� a convergere -> soffre di problemi numerici, percio dobbiamo normalizzare i dati

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
%% Costruzione del problema duale:
% min_{alpha} .5 sum_i sum_j alpha_i alpha_j y_i y_j x_i x_j - sum_i
% alpha_i
%   s.t. 0 <= alpha_i <= C
%        sum_i alpha_i y_i = 0
C=10;
H = diag(Y)*(X*X')*diag(Y); %Corrisponde a:  y_i y_j x_i x_j
% diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));
%n: dimensione problema
% H matrice quadratica
% -ones(n,1): parte lineare (-sum_i alpha_i)
% Y: vincolo Y'alpha=b -> sum_i alpha_i y_i = 0
% zeros(n,1): Lower boundary 0<= alhpa_i
% C*ones(n-1): Upper boundary
% 1.e+8: max iterations
% 1.e-4: what we consider as
% zeros(n,1): aplha iniziale che rispetta i vincoli

w = X'*diag(Y)*alpha;   % w= sum_i alpha_i y_i x_i
%il bias b lo da gi� il problema di ottimizzazione

%% TEST: Sabbia per testare il separatore

ns = 10000;
XS = 2*rand(ns,d)-1;
YS = XS*w+b;
figure; hold on; box on; grid on;
%Plotto il margine


%I punti al di la dell'iperpiano di supporto (non all'interno del margine)
%devono avere alpha = 0
%I punti neri sono quelli che giacciono esattamente sull'iperpiano di
%supporto (quindi per cui 0<alpha<C)
%Se diminuisco C cosa succede? C � una REGOLARIZZAZIONE, se la diminuisco me ne
%frego di fare errori, importante � avere grosso margine -> la dimensione
%del margine aumenta
plot(XS(YS>0,1),XS(YS>0,2),'.c')
plot(XS(YS<0,1),XS(YS<0,2),'.m')
plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
plot(XS(YS<-1,1),XS(YS<-1,2),'.r')
 
plot(X(Y==+1,1),X(Y==+1,2),'ob', 'MarkerSize',8);
plot(X(Y==-1,1),X(Y==-1,2),'or','MarkerSize',8);
 
plot(X(alpha==C,1),X(alpha==C,2),'*g','MarkerSize',8)                 %dentro il margine
plot(X(alpha==0,1),X(alpha==0,2),'*y','MarkerSize',8)                 %fuori dal margine
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k','MarkerSize',8) %sull'iperpiano di supporto


pause();
%% Passiamo ora al non lineare (Kernel)
%costruzione spirale di archimede (problema altamente non lineare)
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

figure; hold on; box on; grid on;
plot(X(Y==+1,1),X(Y==+1,2),'ob')
plot(X(Y==-1,1),X(Y==-1,2),'or')
% se provassimo a lanciare la SVM lineare:

C=1;
H = diag(Y)*(X*X')*diag(Y);
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));

w = X'*diag(Y)*alpha;
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = XS*w+b;
figure; hold on; box on; grid on;
plot(XS(YS>0,1),XS(YS>0,2),'.c')
plot(XS(YS<0,1),XS(YS<0,2),'.m')
plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
plot(XS(YS<-1,1),XS(YS<-1,2),'.r')
plot(X(Y==+1,1),X(Y==+1,2),'ob')
plot(X(Y==-1,1),X(Y==-1,2),'or') 
plot(X(alpha==C,1),X(alpha==C,2),'*g')
plot(X(alpha==0,1),X(alpha==0,2),'*y')
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k')

%Una linea come separatore fa quello che pu�, che in questo caso � molto
%poco

pause();
%%
%Proviamo ora invece ad utilizzare un kernel gaussiano
% con questo kernel non possiamo calcolare i pesi w in quanto non
% conosciamo le nostre features fi(x)

C=1;

%gamma=15;
gamma = 500;
%H = diag(Y)*(X*X')*diag(Y); sostinuiamo con il kernel
H = diag(Y)*exp(-gamma*pdist2(X,X))*diag(Y);
% diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
[~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),Y,zeros(n,1),C*ones(n,1),1e+8,1e-6,zeros(n,1));
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;
figure; hold on; box on; grid on;

plot(XS(YS>0,1),XS(YS>0,2),'.c')
plot(XS(YS<0,1),XS(YS<0,2),'.m')
plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
plot(XS(YS<-1,1),XS(YS<-1,2),'.r')
 
plot(X(Y==+1,1),X(Y==+1,2),'ob')
plot(X(Y==-1,1),X(Y==-1,2),'or')
 
plot(X(alpha==C,1),X(alpha==C,2),'*g','MarkerSize',8)
plot(X(alpha==0,1),X(alpha==0,2),'*y','MarkerSize',8)
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k','MarkerSize',8)

%aumentando il gamma (inversamente proporzionale alla varianza delle gaussiane)
%aumento la non linearit� del mio modello -> ho delle gaussiane pi� strette
%Qui quello che conta � il gamma, se mettiamo gamma = 10 li azzecca tutti
% notiamo inolte che tutti i punti sono neri: ci� significa che tutti i
% punti sono sul margine, ci� � ovvio perch� sto mettendo una gaussiana molto stretta
% ("uno spillo") su % tutti  i punti per cui ogni punto sta esattamente
%sul margine ---- Perch�?????? cosa � il margine usando i kernel???
% Notiamo che aumentando ancora gamma posso vedere le gaussiane con pallini
% azzurri intorno ai sampre questo ci fa vedere le varie gaussan intorno ai
% miei sample.

% il problema di questo affare qui � che io devo settare il gamma ed il C a
% mano, devo trovare quelli giusti

%come faccio a farlo? devo effettuare la MODEL SELECTION
pause();
%% Model Selection -> cerco il miglior C, gamma

%abbiamo due strade migliori: bootstrap e cross validation, usiamo cross
%validation
nk= 10; %numero cross validation
%Nella realt� 10 � molto poco, almeno ci vuole un 100
err_best = +Inf;

%Notiamo che non dobbiamo calcolare le distanze tutte le volte, mi baster�
%farlo una volta all'inizio
Q = pdist2(X,X);
% /!\ Perch� sono partito da soluzioni piccole andando verso soluzioni
% grandi? perch� applico il rasoio di Okkam, cio� a parit� di errore scelgo
% sempre la soluzione pi� semplice (aggiorno il best solo se err > err_best)
% e non >=, err_best � stato ottenuto con parametri pi� piccoli

%Il fatto che sia semplice o vedo perch� che il margine �
%abbastanza grande cio� predico blu esattamente solo quando ci sono tanti
%punti blu (idem per rosso) se non cerco di tenere il margine grande. Se
%invertissi i logspace avrei la soluzione + complessa (a minor errore) x cui
%avrei una figura a minor margine e piu strana
for gamma = logspace(-6,4,30)
    H = diag(Y)*exp(-gamma*Q)*diag(Y);
    for C = logspace(-6,4,30)
        %splitto il dataset
        err = 0;
        for k=1:nk
            i = randperm(n);
            nv = round(.3*n); %30% dei dati per la validation
            UB = C*ones(n,1);
            % mi baster� settare questi indici a 0 per non contarare i punti nel training
            UB(i(1:nv)) = 0;
            %SMO_ab non funziona perch� abbiamo inserito dei vincoli tipo
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

% Notiamo che la soluzione � una soluzione di sicurezza, cio� ha il margine
% molto largo -> predico blu solo esattamente quando ci sono tanti punti
% blu (idem per rosso), se no tento di tenere il margine pi� largo possibile
% questo � perch� abbiamo scelto la soluzione a minimo C, gamma -> pi�
% semplice possibile

%Se vediamo lo spazio bi dimensionale [C, gamma] non ci sar� una soluzione
%ottima, perch� entrambi possono essere usati pi� o meno per regolarizzare, vi sar�
%una linea ottima in tale spazio
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = exp(-gamma*pdist2(XS,X))*diag(Y)*alpha+b;

figure, hold on, box on, grid on
plot(XS(YS>0,1),XS(YS>0,2),'.c')
plot(XS(YS<0,1),XS(YS<0,2),'.m')
plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
plot(XS(YS<-1,1),XS(YS<-1,2),'.r')

% Nota: alcune volte la soluzione non viene corretta, � sensato ci�?
 
plot(X(Y==+1,1),X(Y==+1,2),'ob','markersize',8,'linewidth',8)
plot(X(Y==-1,1),X(Y==-1,2),'or','markersize',8,'linewidth',8)
 
plot(X(alpha==C,1),X(alpha==C,2),'*g','markersize',8,'linewidth',8)
plot(X(alpha==0,1),X(alpha==0,2),'*y','markersize',8,'linewidth',8)
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k','markersize',8,'linewidth',8)
pause();
%% Multiclasse
clear; clc; close all;
 %mettiamod delle palle su una sfera
 
 n=30;
 d=2;
 r=4; % raggio sfera grossa
 c=5; % #classi
 
 X=[]; Y=[];
 j=0; % indicatore classi
 for theta=0:2*pi/c:2*pi-eps
     %passo di 2*pi/c per avere classi egualmente distinte
     X=[X; randn(n,1)+r*cos(theta),randn(n,1)+r*sin(theta)];
     j=j+1;
     Y=[Y; j*ones(n,1)];
     
 end
 [n, d] = size(X);

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
%plotto
colors = 'rbygkmc';
figure, hold on, box on, grid on
for i = 1:c
        plot(X(Y==i,1),X(Y==i,2), ['o' colors(i)],'markersize',10,'linewidth',3);
end

%Applichiamo la tecnica OVO in quanto � molto vantaggiosa rispetto alla
%OVA:
%    - problemi pi� piccoli -> meno dati
%    - classi bilanciate e non sbilanciate di tipo 1vs tanti
%Svantaggi: devo fare n^2/2 classificatori e poi "classificare per votazione di questi"


% per tutte le possibili combinazioni:
W = []; B=[]; %pesi dei vari classificatori
for i = 1:c
    for j=i+1:c
        %qui faccio classificatore di i vs j
        %questi sono i flag che user� per questo classificatore
        fm = Y ==i;  %classe i ("negativa")
        fp = Y == j;    %classe j ("positiva")
        XP = [X(fm,:); X(fp,:)]; %Xproblema prendo solo i dati delle classi i OR j
        YP = [-ones(sum(fm),1); ones(sum(fp),1)];
        n = length(YP);
        C=1; %suppongo C=1;
        H = diag(YP)*(XP*XP')*diag(YP);
        [~,~,alpha,b] = SMO2_ab(n,H,-ones(n,1),YP,zeros(n,1),C*ones(n-1),1.e+8,1.e-4,zeros(n,1));
        w = XP'*diag(YP)*alpha; 
        W = [W,w];
        B = [B,b];
    end
end

ns = 10000;
XS = 2*rand(ns,d)-1;
YS = [];
k = 0;
for i = 1:c
    for j = i+1:c
        k = k + 1;
        %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
        %i o j
        tmp = XS*W(:,k)+B(k);
        tmp(tmp>0) = j;
        tmp(tmp<=0) = i;
        YS = [YS, tmp]; %#ok<AGROW>
    end
end


% per scegliere posso pensare di usare la moda, il valore pi� votato tra i
% vari classificatori che dicono classe i vs classe j
YS = mode(YS,2);

for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.' colors(i)]);
end

%ci accorgiamo che la soluzione non � quella corretta in quanto il problema
%� simmetrico, ha delle simmetrie, ma tutte le linee non convergono allo stesso punto
% SVM per� � consistente perci� all'aumentare del numero dei punti si
% raggiunge il bayesiano
