clear; clc;

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


C=500;
k=0;
for gamma = [1 2 3 5 10 100]
    k=k+1;
    %H = diag(Y)*((X*X').^p)*diag(Y);    
    H = diag(Y)*exp(-gamma*pdist2(X,X,'cityblock'))*diag(Y);
    % diag(Y): sulla diagonale ci sono le Y, dalle altre parti 0
    [~,~,~,alpha,b] = SMO(n,H,-ones(n,1),Y, C*ones(n,1),1e+8,1e-3,zeros(n,1));


    ns = 10000;
    XS = 2*rand(ns,d)-1;
    %YS = ((XS*X').^p)*diag(Y)*alpha+b;
    YS = exp(-gamma*pdist2(XS,X,'cityblock'))*diag(Y)*alpha+b;

    subplot(2,3, k); hold on; box on; grid on;

    plot(XS(YS>0,1),XS(YS>0,2),'.c')
    plot(XS(YS<0,1),XS(YS<0,2),'.m')
    plot(XS(YS>+1,1),XS(YS>+1,2),'.b')
    plot(XS(YS<-1,1),XS(YS<-1,2),'.r')
    plot(X(Y==+1,1),X(Y==+1,2),'ob','markersize',5,'linewidth',5)
    plot(X(Y==-1,1),X(Y==-1,2),'or','markersize',5,'linewidth',5)
    plot(X(alpha==C,1),X(alpha==C,2),'*g','markersize',5,'linewidth',5)
    plot(X(alpha==0,1),X(alpha==0,2),'*y','markersize',5,'linewidth',5)
    plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'*k','markersize',5,'linewidth',5)
    title(['gamma = ' num2str(gamma)])
end