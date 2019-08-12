clear; clc; close all;
% Ultima volta algoritmo KLRS: Kernel regularized least squares
% abbiamo visto come settare gli iperparametri, della regolarizz e del
% kernel
%abbiamo visto come fare model selection ed anche nel caso multiclasse

%Oggi vediamo che possiamo fare il multiclasse in modi diversi: OVA, OVO o
%augmented binary

%% OVO
% Usiamo KLRS solo per brevità del codice, non facciamo model selection
 n=30;
 d=2;
 r=4; % raggio sfera grossa
 c=5; % #classi
 
 X=[]; Y=[];
 j=0; % indicatore classi
 for theta = 0:2*pi/c:2*pi-eps
    j = j + 1;
    X = [X; randn(n,1)+r*cos(theta), randn(n,1)+r*sin(theta)]; %#ok<AGROW>
    Y = [Y; j*ones(n,1)];%#ok<AGROW>
 end
% % Metto la non linearità: Palla esterna:
% j = 0;
%  for theta=0:2*pi/c:2*pi-eps
%      %passo di 2*pi/c per avere classi egualmente distinte
%      X=[X; randn(n,1)+2*r*cos(theta),randn(n,1)+2*r*sin(theta)];
%      j=j+1;
%      Y=[Y; (c+1-j)*ones(n,1)];
%      
%  end
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

%hyperparams (senza fare model selection)
gamma = 1;
lambda = 1;
Q = exp(-gamma*pdist2(X,X));
ALPHA = cell(c*(c-1)/2,1); 
INDEX = cell(c*(c-1)/2,1);
il = (1:n)'; %ora non abbiamo divisioni, perciò considero tutti gli n punti
im = 0;
for i = 1:c
    for j=i+1:c
        im = im+1;
        fm = Y ==i;  %classe i ("negativa")
        fp = Y == j;    %classe j ("positiva")
        ilp = [il(fm); il(fp)];
        YP = [-ones(sum(fm),1); ones(sum(fp),1)];
        alpha = (Q(ilp,ilp) + lambda*eye(length(ilp)))\YP; %pseudo inversa
        ALPHA{im} =alpha; %è un concatenamento con i cell
        %non ricalcolo mai la stessa quantità, utilizzo solamente
        %diverse porzioni della stessa matrice secondo quello che
        %mi serve
        INDEX{im}= ilp;
    end
end

%test:
ns = 10000;
XS = 2*rand(ns,d)-1;
YF = [];
im = 0;
Q = exp(-gamma*pdist2(XS,X));
for i = 1:c
    for j = i+1:c
        im = im + 1;
        tmp = Q(:,INDEX{im}) * ALPHA{im};
        tmp(tmp>0) = j;
        tmp(tmp<=0) = i;
        YF = [YF, tmp]; %#ok<AGROW>
    end
end
YS = mode(YF,2);
colors = 'rbygkmc';
figure, hold on, box on, grid on
for i = 1:c
        plot(X(Y==i,1),X(Y==i,2), ['o' colors(i)],'markersize',10,'linewidth',3);
end
for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.' colors(i)]);
end


%Notiamo che non facendo model selection la soluzione non è ottimale, non è
%simmetrica
pause();
%% OVA

clear; clc;close all;
 n=30;
 d=2;
 r=4; % raggio sfera grossa
 c=5; % #classi
 
X=[]; Y=[];
j=0;
%Palla interna:
for theta = 0:2*pi/c:2*pi-eps
    j = j + 1;
    X = [X; randn(n,1)+r*cos(theta), randn(n,1)+r*sin(theta)]; %#ok<AGROW>
    Y = [Y; j*ones(n,1)];%#ok<AGROW>
 end
% Metto la non linearità: Palla esterna:
j = 0;
 for theta=0:2*pi/c:2*pi-eps
     X=[X; randn(n,1)+2*r*cos(theta),randn(n,1)+2*r*sin(theta)];
     j=j+1;
     Y=[Y; (c+1-j)*ones(n,1)];
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

gamma = 1;
lambda = 1;
ALPHA = []; %matrice che contiene parametri di ogni modello (c modelli)
Q = exp(-gamma*pdist2(X,X));

% nel OVA creo c modelli (c: # classi) ogniuno dei quali fa una classe vs
% le altre
% gli alpha diventano una matrice perche ho bisongo di un vettore di alpha per
%ogni modello cioè c vettori
for i=1:c
    %per ogni problema mi devo creare positivi e negativi, in tutti i
    %problemi ora ho tutti i punti, non devo prendere un sottoinsieme
    % X sarà perciò la stessa X, Y cambierà:
    YP = Y;
    YP(Y==i) =+1; %è importante l'ordine, la prima dobbiamo metterla a -1 per evitare che ci siano degli 1
    % della Y che si confondono, in questo caso non c'è problema perchè
    % usiamo vettori diversi ma volessimo farlo con lo stesso no
    YP(Y~=i) =-1;
    alpha = (Q + lambda*eye(n))\YP;
    %la fase in avanti qui non potrà essere la MODA, perchè abbiamo c
    %classificatori, non ci sarà mai più di 1 classificatore che mi dice
    %"quella giusta è quella" ma dobbiamo considerare la DISTANZA DAL MARGINE:
    % (wx)/||w|| = f(x)/||w||  ove ||w|| = sqrt(alpha*Q*alpha)
    % cosi trasformiamo quello che sarebbe una semplice indicazione del segn
    % data da f(x) in una misura di distanza dal margine cosi da poter vedere
    % quale è + distante dal margine e scegliere quello come migliore
    %classificheremo poi quindi come la classe con la massima distanza dal
    %margine
    alpha = alpha/sqrt(alpha'*Q*alpha); %normalizziamo subito gli alpha per confrontare le distanze dal margine
    
    ALPHA= [ALPHA, alpha]; %accosto agli altri parametri la colonna relativa a questo classificatore
end
% vedi articolo [platt probabilistic output for SVM] (parag 2.2)
% come trasformare f(x) = wx in una probabilità di appartenere ad una classe -> applico sigmoide fatta
% con un parametro alpha da calcolare tenendosi parte dei dati

%FASE IN AVANTI:
ns = 10000;
XS = 2*rand(ns,d)-1;
YF = exp(-gamma*pdist2(XS,X))*ALPHA; %su igni riga ho classificazione di ciascun classificatore
% riga: [ | | | | ] ove nella colonna i vi è il risultato del
% classificatore i vs ALL, ->
%vi possono essere dei casi in cui più di una è positiva o tutte son
%negative ma se prendo l'indice del massimo trovo la
% classe che ha predetto con massimo margine
[~, YS] = max(YF,[],2);
colors = 'rbygkmc';
figure, hold on, box on, grid on
for i = 1:c
        plot(X(Y==i,1),X(Y==i,2), ['o' colors(i)],'markersize',10,'linewidth',3);
end
for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.' colors(i)]);
end
title('Con kernel');

% Con separatore lineare: cambia solo Q e fase in avanti
gamma = 1;
lambda = 1;
ALPHA = [];
Q = X*X'; % <- CASO SENZA KERNEL
for i=1:c
    YP = Y;
    YP(Y==i) =+1;
    YP(Y~=i) =-1;
    alpha = (Q + lambda*eye(n))\YP;
    alpha = alpha/sqrt(alpha'*Q*alpha);
    ALPHA= [ALPHA, alpha];
end
%FASE IN AVANTI:
ns = 10000;
XS = 2*rand(ns,d)-1;
YF = (XS*X')*ALPHA; % <- CASO SENZA KERNEL
[~, YS] = max(YF,[],2);
colors = 'rbygkmc';
figure, hold on, box on, grid on
for i = 1:c
        plot(X(Y==i,1),X(Y==i,2), ['o' colors(i)],'markersize',10,'linewidth',3);
end
for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.' colors(i)]);
end
title('Senza kernel');
%Notiamo come con un separatore lineare, la soluzione viene molto bella con
%i parametri scelti (molto simmetrico)
%notiamo che però tornando al caso di dati non lineari la soluzione fa
%schifo
%OVA è molto più semplice da implementare di OVO ed è più intuitivo,
%risolvendo meno problemi di ottimizzazione
% lo svantaggio è che questi problemi sono più grandi
pause();
%% Augmented binary
clear; clc; close all;
n=30;
 d=2;
 r=4; % raggio sfera grossa
 c=5; % #classi
 
X=[]; Y=[];
j=0;
%Palla interna:
for theta = 0:2*pi/c:2*pi-eps
    j = j + 1;
    X = [X; randn(n,1)+r*cos(theta), randn(n,1)+r*sin(theta)]; %#ok<AGROW>
    Y = [Y; j*ones(n,1)];%#ok<AGROW>
 end
% Metto la non linearità: Palla esterna:
j = 0;
 for theta=0:2*pi/c:2*pi-eps
     X=[X; randn(n,1)+2*r*cos(theta),randn(n,1)+2*r*sin(theta)];
     j=j+1;
     Y=[Y; (c+1-j)*ones(n,1)];
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

gamma = 1;
lambda = 1;

%AUGMENTED BINARY: ogni sample viene raddoppiato c volte dalla parte delle
%x, affiancando loro una matrice cxc fatta in maniera diagonale diag(1)
% le y saranno 1 per il sample del nuovo dataset ove x è affiancato con l'1
% nella posizione della giusta classe

XAB = [];
YAB = [];
%Costruzione matrici
for i= 1:c
    B = zeros(n,c); B(:,i) = 1;
    XAB = [XAB; [X, B]];
    YAB = [YAB; (Y==i)-(Y~=i)];
    %ogni volta aggiungo la matrice (X affiancata a B) alle 
    %alle Y ogni volta affianco il valore {1,-1} per vedere dove Y è uguale
    %o meno alla classe i
end

% Risolvo il problema: un solo problema BICLASSE, che poi distingueremo a
% seconda della matrice B (un modello molto grosso)
%Q = XAB*XAB'; % <- CASO SENZA KERNEL (non usabile con augmented binary)
Q = exp(-gamma*pdist2(XAB,XAB));
alpha = (Q + lambda*eye(n*c))\YAB;

%Fase in avanti:
%ogni punto viene mappato aggiungendo una riga [0 0 0  .. 1 0 0] ove l'1 è
%la classe che vogliamo vedere se vera o falsa
ns = 10000;
XS = 2*rand(ns,d)-1;
XSAB = [];
for i= 1:c
    B = zeros(ns,c); B(:,i) = 1;
    XSAB = [XSAB; [XS, B]];
end
%YF = (XSAB*XAB')*alpha; % <- CASO SENZA KERNEL (non usabile con augmented binary)
YF = exp(-gamma*pdist2(XSAB,XAB))*alpha;
%vorrei che i risultati del sample i fossero affiancati su una stessa riga
%(in matlab le matrici vengono riempite per colonna -> in un reshape prima riempio (1,1) poi (2,1) (3,1)...)
YF = reshape(YF,ns,c);
%dopo di che come prima classifico secondo il massimo rispetto ad una
%colonna
%(rispetto a OVA non abbiamo normalizzato per ||w|| per trovare la
%distanza perchè non funzione di separazione è sempre la stessa)
[~, YS] = max(YF, [],2);
colors = 'rbygkmc';
figure, hold on, box on, grid on
for i = 1:c
        plot(X(Y==i,1),X(Y==i,2), ['o' colors(i)],'markersize',10,'linewidth',3);
end
for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.' colors(i)]);
end
%Notiamo che nel caso lineare l'augmented binary NON FUNZIONA perchè se
%faccio f(x) = [w_x w_b][x 0 0 ... 1 .. 0] e poi prendo il massimo la
%classificazione NON dipenderà più da x in quanto la parte w_x*x sarà
%uguale per tutti

%%
% Nella pratica nessuno utilizza mai augmented binary, nella teoria lo
% usano tutti perchè è utile per trovare risultati (ci riconduciamo al caso biclasse)
% tutti usano OVA [in defence of one vs all] si dimostra che è uguale
% all'OVO
