
%% Problema della minimum Closing Ball attorno ai punti
clear; clc; close all;

%Vediamo cosa servono i moltiplicatori di Lagrange
% Vediamo i kernel senza regolarizzazione

n = 100;
d = 2;
c = 10*ones(1,d); %centro delle distribuzioni
a = 8; %range del plot
X = randn(n,d) + repmat(c,n,1); %Gaussiana centrata nel punto c

%Butto in questo range un po' di sabbia per poi vedere come è fatto il mio
%separatore
ns = 10000; %numero punti sabbia
XS = rand(ns,d)*2*a; %distribuzione random larga 2a -> dimensione del plot
XS = XS + repmat([c(1)-a, c(2)-a],ns,1); %aggiungo bias per spostarli sul nostro plot

figure; hold on, box on, grid on;
plot(X(:,1),X(:,2),'ob');
xlim([c(1)-a, c(1)+a]);
ylim([c(2)-a, c(2)+a]);

% Se noi vogliamo trovare il minimo degli errori di classificazione, questo
% è un problema non convesso cioè la funzione errore è 1 -1(yf(x)) ->
% l'algoritmo del percettrone risolve questo problema nell'ipotesi in cui
% le classi siano lineramente separabili, cioè nella regione a sx, ove la
% funzione errore è = 0 -> è convessa

%Algoritmo:
% min alpha' Q alpha - diag(Q)'*alpha
%       a := centro della closinga ball
% s.t.  sum(alpha) = 1
%       alpha > 0
%       a = sum_i alpha_i x_i
%       R -> alpha_i >0 ||x_i - a || = R

% Nota: rispetto a quanto fatto a lezione questo risolve il problema senza
% considerare l'esistenda di punti al di fuori della sfera, 
% non è quindi ancora un algoritmo di Outlayer Detection

% questo è un probema di minimizzazione quadratica, vi sono un sacco di
% metodi per minimizzarlo
% in matlab quadprog -> help quadprog
% min 0.5*x'*H*x + f'*x   subject to:  A*x <= b 
% questo però fa un po schifo perchè è generale ed a volte non converge
% usiamo SMO; Sequential Minimal Optimization
% Una forma quadratica può essere minimizzata un parametro alla volta,
% scendo dalla conca pian piano una variabile alla volta
% in questo problema però abbiamo un vincolo lineare, con un vincolo
% lineare non possiamo cambiare una sola variabile alla volta perchè il
% vincolo lineare deve essere soddisfatto, perciò devo cambiare almeno due
% variabili alla volta, con n vincoli lineari, n+1 variabili alla volta

% function [Nabla,err,x,bias] = SMO2_ab(n,H,f,a,LB,UB,maxiter,eps,alpha_s)
    % min_{x} .5 x H x + f' x 
    %         LB <= x <= UB
    %         a' x = b
    % n         grandezza problema length(x)
    % maxiter   max num it
    % eps       precisione -> zero numerico
    % alpha_s   punto di inizio valido per x
    % -- Output:
    % Nabla     ....
    % err       flag di ok, indica se l'algoritmo non è arrivato a convergenza
    % x         valore della soluzione ottima
    % bias      moltiplicatori di Lagrange del primale

% notiamo che l'algoritmo SMO non prende b in ingresso dei vincoli lineari a' x = b
% questo perchè la x iniziale deve essere tale che tale vincolo deve essere
% soddisfatto, e quindi si può ritrovare b

%(Il nostro problema di minimizzazione è sulla variabile alpha, non x)



Q = X*X';
f = -diag(Q);
[~,~,alpha,~] = SMO2_ab(n,2*Q,f,ones(n,1),zeros(n,1),Inf*ones(n-1),100000,1.e-4,[1;zeros(n-1,1)]);

%ottimiziamo 2*Q perchè l'algoritmo ottimizza 1/2*Q
% un punto d'inizio valido per alpha è [1 0 0 .. 0] in quanto tutti i miei vincoli
% sono soddisfatti, la somma fa 1

a = X'*alpha; %centro della sfera
%Per trovare il raggio
%in una sfera in 2 dimensioni ci devono essere almeno 2 punti (in generale
%3) punti sul bordo
% I punti sul bordo della sfera hanno i vincoli del primale che e hannno
% alpha > 0 -> possiamo ricavarci il raggio come la
% distanza tra uno di questi ed il centro
i = find(alpha > 0);
i = i(1); %Prendo il primo, a caso
R = norm(X(i,:)' -a);
plot(a(1), a(2), '+r', 'MarkerSize',10);

%disegnamo la sfera usando la sabbia
d = pdist2(XS,a'); %distanza dei punti di XS da a
d = d>R;      %considero solo quelli fuori dal cerchio
plot(XS(d,1),XS(d,2),'.k');

plot(X(alpha>0,1),X(alpha>0,2),'*g')
%Con le condizioni di KKT pur non sapendo se l'ottimizzazione ha funzionato
%o no posso vedere se questa è buona:
% - posso vedere se tutti i punti stanno dentro la sfera
% - posso vedere se tutti i punti corrispondenti ad alpha_i > 0 stanno
% esattamente sulla sfera
% cioè andiamo a vedere se tutte le kkt sono soddisfatte con precisione
% abbastanza buona


pause();
%% Cosa faccio se il problema non è più lineare? cioè se i dati non hanno dimensione sferica?
clc; clear; close all;
%Ora facciamo due palle di punti

n = 100;
d = 2;
c = 5*ones(1,d); %centro delle distribuzioni
a = 14; %range del plot

X = [randn(n/2,d) + repmat(c,n/2,1); ...
    randn(n/2,d) - repmat(c,n/2,1)];%Gaussiana centrata nel punto c

ns = 10000; %numero punti sabbia
XS = rand(ns,d)*2*a; %distribuzione random larga 2a -> dimensione del plot
XS = XS + repmat([-a,-a],ns,1); %aggiungo bias per spostarli sul nostro plot
figure, hold on, box on, grid on
plot(X(:,1),X(:,2),'ob')
xlim([-a,+a])
ylim([-a,+a])
 

Q = X*X';
f = -diag(Q);
[~,~,alpha,~] = SMO2_ab(n,2*Q,f,ones(n,1),zeros(n,1),Inf*ones(n-1),100000,1.e-4,[1;zeros(n-1,1)]);
a = X'*alpha; %centro della sfera
i = find(alpha > 0);
i = i(1);
R = norm(X(i,:)' -a);
plot(a(1), a(2), '+r', 'MarkerSize',10);

d = pdist2(XS,a'); %distanza dei punti di XS da a
d = d>R;      %considero solo quelli fuori dal cerchio
plot(XS(d,1),XS(d,2),'.k');

plot(X(alpha>0,1),X(alpha>0,2),'*g')

%Se ora lanciassi l'algoritmo di prima non va bene, perchè i miei dati non
%sono distribuiti secondo quel modo, dovrebbero venire due palle ed invece
%ne viene una sola

% -------- METODO NON LINEARE

%Trasformo il mio problema di ottimizzazione:
%usando il kernel di xi xj

% min alpha' Q alpha - diag(Q)'*alpha
% s.t.  sum(alpha) = 1
%       alpha > 0
%       Q_ji = K(xi,xj)

%       R -> alpha_i >0 ||x_i - a || = R
%       ove il modulo si calcola come:
%       |a - x|^2 = sum_i sum_j ai aj K(x_i, x_j) -2sum_i alpha_i + K(x_i, x)
%       +K(x,x)
%       Il primo termine è una costante per tutti i punti e posso toglierlo 
%       -> LO TOLGO DA TUTE LE PARTI, sia nel calcolo di R che in quello di qualsiasi
%       di qualsiasi altra distanza
%       il secondo con il kernel gaussiano è = 1


a = 14;
%gamma = 0.03451; % sdt dev gaussiana
gamma = 0.00318; % sdt dev gaussiana

figure; hold on; box on; grid on;
plot(X(:,1),X(:,2),'ob')
xlim([-a,+a])
ylim([-a,+a])

%Usiamo un kernel gaussiano
Q = exp(-gamma*pdist2(X,X));
f = -diag(Q);
[~,~,alpha,~] = SMO2_ab(n,2*Q,f,ones(n,1),zeros(n,1),Inf*ones(n-1),100000,1.e-6,[1;zeros(n-1,1)]);
%Ora non possiamo più trovare il centro della distrib a
i = find(alpha > 0);
i = i(1);
%abbiamo tolto dai calcoli, sia di R che di d il termine a'a che intanto è
%constante per tutti i punti

R = Q(i,i) - 2*Q(i,:)*alpha; %||x_i - a || = R

d = 1-2*exp(-gamma*pdist2(XS,X))*alpha; %|a - x|^2 con K(x,x) = 1
d = d>R;      %considero solo quelli fuori dal cerchio
plot(XS(d,1),XS(d,2),'.k');

plot(X(alpha>0,1),X(alpha>0,2),'*g')

% Ulteriori osservazioni:
% - Man mano che metto gamma più piccolo la superficie diventa più ovale(DIREI)
% - Se gamma cresce invece la superficie si rimpicciolisce (Q è una misura
% della distanza tra i punti, se gamma cresce sto aumentando le distanze CREDO)


