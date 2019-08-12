function [Nabla,err,x,bias] = SMO2_ab(n,H,f,a,LB,UB,maxiter,eps,alpha_s)
    % min_{x} .5 x H x + f' x 
    %         LB <= x <= UB
    %         a' x = b
    % n         grandezza problema length(x)
    % maxiter   max num it
    % eps       precisione
    % alpha_s   punto di inizio valido per x
    % Nabla     ....
    % err       flag di ok
    % x         valore della soluzione ottima
    % bias      moltiplicatori Lagrange del primale NON NE SONO SICURO
    x = alpha_s;
    Nabla = f;
    for i = 1:n
        if (x(i) ~= 0.0)
            for j = 1:n
                Nabla(j) = Nabla(j) + H(j,i) * x(i);
            end
        end
    end    
    iter = 0;
    while (true)   
        minF_up  =  Inf;
        maxF_low = -Inf;
        for i = 1:n   
            F_i = Nabla(i)/a(i);
            if(LB(i) < x(i) && x(i) < UB(i)) 
                if (minF_up > F_i)  
                    minF_up = F_i; 
                    u = i; 
                end
                if (maxF_low < F_i) 
                    maxF_low = F_i; 
                    v = i; 
                end 
            elseif((a(i) > 0 && x(i) == LB(i)) || (a(i) < 0 && x(i) == UB(i))) 
                if (minF_up > F_i)
                    minF_up = F_i; 
                    u = i;
                end
            elseif((a(i) > 0 && x(i) == UB(i)) || (a(i) < 0 && x(i) == LB(i))) 
                if (maxF_low < F_i)
                    maxF_low = F_i; 
                    v = i; 
                end
            end
        end        
        if(maxF_low - minF_up <= eps )
            err = 0.0;
            break
        end
        iter = iter + 1;
        if(iter >= maxiter)
            err = 1.0;
            break
        end        
        if (a(u) > 0) 
            tau_lb = (LB(u)-x(u))*a(u); 
            tau_ub = (UB(u)-x(u))*a(u); 
        else
            tau_ub = (LB(u)-x(u))*a(u); 
            tau_lb = (UB(u)-x(u))*a(u);
        end
        if (a(v) > 0)
            tau_lb = max(tau_lb,(x(v)-UB(v))*a(v)); 
            tau_ub = min(tau_ub,(x(v)-LB(v))*a(v)); 
        else
            tau_lb = max(tau_lb,(x(v)-LB(v))*a(v)); 
            tau_ub = min(tau_ub,(x(v)-UB(v))*a(v)); 
        end
        tau = (Nabla(v)/a(v)-Nabla(u)/a(u))/...
              (H(u,u)/(a(u)*a(u)) + H(v,v)/(a(v)*a(v)) - 2*H(v,u)/(a(u)*a(v)));
        tau = min(max(tau,tau_lb),tau_ub);
        x(u) = x(u) + tau/a(u);
        x(v) = x(v) - tau/a(v);
        for i = 1:n 
            Nabla(i) = Nabla(i) + H(u,i)*tau/a(u) - H(v,i)*tau/a(v);
        end
    end  
    tsv = 0;
    bias = 0.0;
    for k = 1:n
        if ((x(k) > LB(k)) && (x(k) < UB(k)))
            bias = bias - Nabla(k)/a(k);
            tsv = tsv + 1;
        end
    end
    if (tsv > 0)
        bias = bias / tsv;
    else       
        bias = -(maxF_low + minF_up) / 2.0;
    end
end