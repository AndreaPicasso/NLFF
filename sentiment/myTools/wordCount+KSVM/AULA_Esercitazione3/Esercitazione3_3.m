clear; clc; close all;
%La volta precedente abbiamo fatto SVM ed abbiamo controllato che tutte le
%condizioni KKT fossero soddisfatte
%Dopodichè siamo passati all'SVM non linare, gaussiana, perchè questo può
%imparare qualsiasi dataset con opportuni valori di C e gamma
%complete cross validation: si utilizzano tutte le possibili combinazioni di k elementi
%con n sample

% - Capire come fare la model selection nel caso di problemi multiclasse
% - come implementare altre strategie OVA e Augmented Bynary


%% Model selection caso multiclasse
% potremmo trovare gli iperparametri separatamente per ogni classificatore OVO
% ma questa NON è la giusta soluzione, a noi interessa il problema completo,
% non vogliamo ottenere i parametri che ottimizzino singolarmente le classi

%(Problema esercizio precedente)
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


%Applichiamo la tecnica OVO
nC = 30; %numero di C da provare
nK = 10; % quanti split di cross validation fare
err = ones(nC,1);
ie = 0; % indice errore per i vari C
CC = logspace(-6,4,nC); %possibili valori di C
%Nota: SVM lineare, l'unico parametro che c'è è C
for C = CC
    ie = ie+1;
    for k=1:nK %k split diversi sui dati
        ip = randperm(n); %sarebbe opportuno dividere fuori dalla cross val. il dataset
        nl = round(.7*n); % percentuale training
        XL = X(ip(1:nl),:); YL = Y(ip(1:nl),:); %training set
        XV = X(ip(nl+1:end),:); YV = Y(ip(nl+1:end),:);

        W = []; B=[]; %pesi dei vari classificatori
        %i j t.c. matrice W triangolare superiore, classifichiamo solo i vs j
        % non i vs i oppure j vs i 
        for i = 1:c
            for j=i+1:c
                %qui faccio classificatore di i vs j
                %questi sono i flag che userò per questo classificatore
                fm = YL ==i;  %classe i ("negativa")
                fp = YL == j;    %classe j ("positiva")
                XP = [XL(fm,:); XL(fp,:)]; %Xproblema prendo solo i dati delle classi i OR j
                YP = [-ones(sum(fm),1); ones(sum(fp),1)];
                npp = length(YP);
                H = diag(YP)*(XP*XP')*diag(YP);
                [~,~,alpha,b] = SMO2_ab(npp,H,-ones(npp,1),YP,zeros(npp,1),C*ones(npp-1),1.e+8,1.e-4,zeros(npp,1));
                w = XP'*diag(YP)*alpha; 
                W = [W,w];
                B = [B,b]; %per aumentare le performance potremmo inizializzarle , tanto sappiamo le dimensioni
            end
        end
        %i j t.c. matrice W triangolare superiore, classifichiamo solo i vs j
        % non i vs i oppure j vs i 
        YF = [];
        im = 0;
        for i = 1:c
            for j = i+1:c
                im = im + 1;
                %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                %i o j
                tmp = XV*W(:,im)+B(im);
                tmp(tmp>0) = j;
                tmp(tmp<=0) = i;
                YF = [YF, tmp]; %#ok<AGROW>
            end
        end
        YF = mode(YF,2);
        err(ie) = err(ie) + mean(YF ~=YV)/nK;
    end
end
[~, i] = min(err);% prendo minimo valore errore
% qui non serve usare il rasoio di okkam, l'errore è un errore medio
% è float, è difficile che siano uguali
C = CC(i); %C ottimo

%Training con il C ottimo
W = [];
B = [];
for i = 1:c
    for j = i+1:c
        fm = Y == i;
        fp = Y == j;
        XP = [X(fm,:); X(fp,:)];
        YP = [-ones(sum(fm),1); ones(sum(fp),1)];
        n = length(YP);
        C = 1;
        H = diag(YP)*(XP*XP')*diag(YP);
        [~,~,alpha,b] = ...
            SMO2_ab(n,H,-ones(n,1),YP,zeros(n,1),C*ones(n,1),1e+8,1e-4,zeros(n,1));
        w = XP'*diag(YP)*alpha;
        W = [W, w]; %#ok<AGROW>
        B = [B, b]; %#ok<AGROW>
    end
end


%Plotto con la sabbia
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = [];
k = 0;
for i = 1:c
    for j = i+1:c
        k = k + 1;
        tmp = XS*W(:,k)+B(k);
        tmp(tmp>0) = j;
        tmp(tmp<=0) = i;
        YS = [YS, tmp]; %#ok<AGROW>
    end
end
YS = mode(YS,2);
colors = 'rbygkmc';
figure, hold on, box on, grid on
for i = 1:c
        plot(X(Y==i,1),X(Y==i,2), ['o' colors(i)],'markersize',10,'linewidth',3);
end
for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.' colors(i)]);
end

figure; plot(1:nC,err);
%se plottiamo err notiamo che l'errore è convesso, prima facciamo
%underfitting, con C più grandi faccio poi overfitting (lo vediamo avendo
% 30 punti, con molti punti è più difficile fare overfitting)

%notiamo inoltre che plottando i dati il centro viene più simmetrico
%rispetto a prima e il bayesiano sembra più vicino dato che il problema ha
%simmetrie
%notiamo inoltre che aumentando il numero di loop di cross validation
%l'errore diventa più stabile

pause();
%% Multiclasse Non Lineare -> due palle concentriche di punti
%Applichiamo ancora SVM lineare, l'unico parametro che c'è è C (nella non lineare
% ci sono anche i parametri del kernel usato)

clear; clc; close all;
 n=30;
 d=2;
 r=4; % raggio sfera grossa
 c=5; % #classi
 
 X=[]; Y=[];
 j=0; % indicatore classi
 %Palla interna:
 for theta = 0:2*pi/c:2*pi-eps
    j = j + 1;
    X = [X; randn(n,1)+r*cos(theta), randn(n,1)+r*sin(theta)]; %#ok<AGROW>
    Y = [Y; j*ones(n,1)];%#ok<AGROW>
 end
% Palla esterna:
j = 0;
 for theta=0:2*pi/c:2*pi-eps
     %passo di 2*pi/c per avere classi egualmente distinte
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
% se ora provo classificatore lineare non imparerà una mazza



%Applichiamo la tecnica OVO
nC = 20;
nK = 10;
err = ones(nC,1);
ie = 0;
CC = logspace(-6,3,nC);
for C = CC
    ie = ie+1;
    for k=1:nK %k split diversi sui dati
        fprintf('%f %f\n',C,k);
        ip = randperm(n);
        nl = round(.7*n); % percentuale training
        XL = X(ip(1:nl),:); YL = Y(ip(1:nl),:);
        XV = X(ip(nl+1:end),:); YV = Y(ip(nl+1:end),:);

        W = []; B=[]; %pesi dei vari classificatori
        %i j t.c. matrice W triangolare superiore, classifichiamo solo i vs j
        % non i vs i oppure j vs i 
        for i = 1:c
            for j=i+1:c
                fm = YL ==i;
                fp = YL == j; 
                XP = [XL(fm,:); XL(fp,:)];
                YP = [-ones(sum(fm),1); ones(sum(fp),1)];
                npp = length(YP);
                H = diag(YP)*(XP*XP')*diag(YP);
                [~,~,alpha,b] = SMO2_ab(npp,H,-ones(npp,1),YP,zeros(npp,1),C*ones(npp-1),1.e+8,1.e-4,zeros(npp,1));
                w = XP'*diag(YP)*alpha; 
                W = [W,w];
                B = [B,b];
            end
        end
        %i j t.c. matrice W triangolare superiore, classifichiamo solo i vs j
        % non i vs i oppure j vs i 
        YF = [];
        im = 0;
        for i = 1:c
            for j = i+1:c
                im = im + 1;
                %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                %i o j
                tmp = XV*W(:,im)+B(im);
                tmp(tmp>0) = j;
                tmp(tmp<=0) = i;
                YF = [YF, tmp]; %#ok<AGROW>
            end
        end
        YF = mode(YF,2);
        err(ie) = err(ie) + mean(YF ~=YV)/nK;
    end
end
[~, i] = min(err);% prendo minimo vaore errore
% qui non serve usare il rasoio di okkam, l'errore è un errore medio
% è float, è difficile che siano uguali
C = CC(i); %C ottimo

%Training con il C ottimo
W = [];
B = [];
for i = 1:c
    for j = i+1:c
        fm = Y == i;
        fp = Y == j;
        XP = [X(fm,:); X(fp,:)];
        YP = [-ones(sum(fm),1); ones(sum(fp),1)];
        n = length(YP);
        C = 1;
        H = diag(YP)*(XP*XP')*diag(YP);
        [~,~,alpha,b] = ...
            SMO2_ab(n,H,-ones(n,1),YP,zeros(n,1),C*ones(n,1),1e+8,1e-4,zeros(n,1));
        w = XP'*diag(YP)*alpha;
        W = [W, w]; %#ok<AGROW>
        B = [B, b]; %#ok<AGROW>
    end
end

%Plotto con la sabbia
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = [];
k = 0;
for i = 1:c
    for j = i+1:c
        k = k + 1;
        tmp = XS*W(:,k)+B(k);
        tmp(tmp>0) = j;
        tmp(tmp<=0) = i;
        YS = [YS, tmp]; %#ok<AGROW>
    end
end
YS = mode(YS,2);
colors = 'rbygkmc';
figure, hold on, box on, grid on
for i = 1:c
        plot(X(Y==i,1),X(Y==i,2), ['o' colors(i)],'markersize',10,'linewidth',3);
end
for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.' colors(i)]);
end

%Se si applica il modello sbagliato non si imparerà mai nulla, queste
%classi sono classificate a seconda della maggioranza dei punti in quello
%spicchio di area
%La figura risultante è composta come: dividiamo lo spazio R^2 in tanti
%settori dati dai vari SVM OVO, per ogni settore prediciamo in base alla
%moda dei vari classificatori
%notiamo che la classe blu (entrambe le palle) non è linearmente separabile
%dalla classe verde, come la classe rossa non è linearmente separabile
%dalla classe nera, in queste aree si crea del macello


%% Minimi quadrati con kernel gaussiano invece che SVM
% Come da appunti scritti a mano, f(x)=wx w = sum(alpha_i x_i)
%  -> min ||Q*alpha-y||^2 + lambda*alpha*Q*alpha
%       f(x) = sum(alpha_i*x_i*x)
%

clear; clc; close all;
 n=30;
 d=2;
 r=4; % raggio sfera grossa
 c=5; % #classi
 
 X=[]; Y=[];
 j=0; % indicatore classi
 %Palla interna:
 for theta = 0:2*pi/c:2*pi-eps
    j = j + 1;
    X = [X; randn(n,1)+r*cos(theta), randn(n,1)+r*sin(theta)]; %#ok<AGROW>
    Y = [Y; j*ones(n,1)];%#ok<AGROW>
 end
% Palla esterna:
j = 0;
 for theta=0:2*pi/c:2*pi-eps
     %passo di 2*pi/c per avere classi egualmente distinte
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

nK= 30;
err_best = Inf;
QT = pdist2(X,X); %distanze, calcolate una volta all'inizio
%for sul primo parametro, gamma
for gamma = logspace(-6,4,30)
    Q = exp(-gamma*QT);
    %notiamo che ho scelto di sprecare memoria rispetto a calcolare
    %all'inizio Q2 così, e dopo annullare l'operazione con
    % Q = log(Q)/gamma
    %for sul secondo parametro
    for lambda = logspace(-6,4,30)
        %cross validation
        err = 0;
        for k = 1:nK
            %spezzo in train e test
            %sarebbe meglio farlo prima del ciclo e usare poi diversi split
            i = randperm(n)';
            nl = round(.7*n);
            %XL = X(ip(1:nl),:); YL = Y(ip(1:nl),:); %training set
            %XV = X(ip(nl+1:end),:); YV = Y(ip(nl+1:end),:);
            % Voglio evitare di dividere i set
            il = i(1:nl);
            iv = i(nl+1:end);
            ALPHA=cell(c*(c-1)/2,1);
            INDEX =cell(c*(c-1)/2,1); %indici dei punti di train
            %visto che conosciamo il numero di classificatori possiamo
            %inizializzare ALPHA e INDEX per farlo andare più veloce
            %cell perchè ogni index ed ogni alpha ha dimensione diversa
            im = 0;
            for i = 1:c
                for j=i+1:c
                    im = im+1;
                    %qui faccio classificatore di i vs j
                    %questi sono i flag che userò per questo classificatore
                    fm = Y(il) ==i;  %classe i ("negativa") solo quelle del LEARNING
                    fp = Y(il) == j;    %classe j ("positiva") solo quelle del LEARNING
                    % poi uccido tutte le altre
                    ilp = [il(fm); il(fp)]; %per questo problema prendo solo gli indici
                    % del training set (il) per cui i label siano i o j
                    YP = [-ones(sum(fm),1); ones(sum(fp),1)];
                    %vogliamo riusare i Q senza spezzare la matrice, senza
                    %spezzare in train e test ma semplicmente considerando le
                    %distanze Q solo per i sample del training set che abbiano
                    % classe i o j
                    alpha = (Q(ilp,ilp) + lambda*eye(length(ilp)))\YP; %pseudo inversa
                    ALPHA{im} =alpha; %è un concatenamento con i cell
                    %non ricalcolo mai la stessa quantità, utilizzo solamente
                    %diverse porzioni della stessa matrice secondo quello che
                    %mi serve
                    INDEX{im}= ilp;
                end
            end
            YF = [];
            im = 0;
            for i = 1:c
                for j = i+1:c
                    im = im + 1;
                    %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
                    %i o j
                    tmp = Q(iv,INDEX{im}) * ALPHA{im}; %train solo su iv
                    tmp(tmp>0) = j;
                    tmp(tmp<=0) = i;
                    YF = [YF, tmp]; %#ok<AGROW>
                end
            end
            YF = mode(YF,2);
            err = err + mean(YF ~=Y(iv));

        end
        if(err_best > err) %rasoio di okkam (NO perchè err è medio, float)
            err_best = err;
            gamma_best = gamma;
            lambda_best = lambda;
        end
       fprintf('%e %e %e %e\n',gamma,lambda,err_best,err);
    end
end

gamma = gamma_best;
lambda = lambda_best;

%Visualizziamo il modello best
%learning:
Q = exp(-gamma*pdist2(X,X)); 
ALPHA = cell(c*(c-1)/2,1); 
INDEX = cell(c*(c-1)/2,1);
im = 0;
il = (1:n)';
for i = 1:c
    for j = i+1:c
        im = im + 1;
        fm = Y == i; fp = Y == j;
        ilp = [il(fm); il(fp)];
        YP = [-ones(sum(fm),1); ones(sum(fp),1)];
        alpha = (Q(ilp,ilp)+lambda*eye(length(ilp)))\YP;
        ALPHA{im} = alpha; 
        INDEX{im} = ilp;
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
        %classifico tra -1,1 in tmp, ma poi lo devo ritrasformare se classe
        %i o j
        tmp = Q(:,INDEX{im}) * ALPHA{im}; %train solo su iv
        tmp(tmp>0) = j; %importante l'ordine tra questa istr e quella dopo PERCHE???
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

%Notiamo che il separatore è buono, tra tutte le gaussiane mette delle
%linee separatrici come farebbe una bayesiana (ovviamente non è perfetta)