function [G,iter,err,alpha,bias] = SMO(np,q,r,y,C,maxiter,eps,alpha_s)

    alpha = alpha_s;
    iter = 0;
    G = r;
    for i = 1:np
        for j = 1:np
            G(j) = G(j) + q(j,i) * alpha(i);
        end
    end
    
    while (1)
        G1 = -Inf;
        G2 = -Inf;
        for i = 1:np
            if(y(i) > 0)
                if(alpha(i) < C(i))
                    if(-G(i) > G1)
                        G1  = -G(i);
                        iG1 = i;
                    end
                end
                if(alpha(i) > 0)
                    if(G(i) > G2)
                        G2  = G(i);
                        iG2 = i;
                    end
                end
            else
                if(alpha(i) < C(i))
                    if(-G(i) > G2)
                        G2  = -G(i);
                        iG2 = i;
                    end
                end
                if(alpha(i) > 0)
                    if(G(i) > G1)
                        G1  = G(i);
                        iG1 = i;
                    end
                end
            end
        end
        
        if ((G1 + G2) < eps)
            err = 0;
            break;
        end
        if((maxiter > 0) && (iter > maxiter))
            err = 5;
            return;
        end

        i = iG1;
        j = iG2;
        iter = iter + 1;
        
        oldAi = alpha(i);
        oldAj = alpha(j);
        
        if(y(i) ~= y(j))
            qq = q(i,i) + q(j,j) + 2 * q(i,j);
            if (qq <= 2.2204e-016)
                if ((-G(i)-G(j)) >= 0.0)
					delta = abs(max([C(i),C(j)]));
				else
					delta = -abs(max([C(i),C(j)]));	
                end
            else
                delta = (-G(i)-G(j)) / qq;
            end
            alpha(i) = alpha(i) + delta;
            alpha(j) = alpha(j) + delta;
            d        = alpha(i) - alpha(j);
            if(d > 0)
                if(alpha(j) < 0)
                    alpha(j) = 0;
                    alpha(i) = d;
                end
            else
                if(alpha(i) < 0)
                    alpha(i) = 0;
                    alpha(j) = -d;
                end
            end
            if(d > (C(i)-C(j)))
                if(alpha(i) > C(i))
                    alpha(i) = C(i);
                    alpha(j) = C(i) - d;
                end
            else
                if(alpha(j) > C(j))
                    alpha(j) = C(j);
                    alpha(i) = C(j) + d;
                end
            end
        else
            qq = q(i,i) + q(j,j) - 2*q(i,j);
            if (qq <= 2.2204e-016)
                if ((G(i)-G(j)) >= 0)
					delta = abs(max([C(i),C(j)]));
				else
					delta = -abs(max([C(i),C(j)]));
                end
            else
                delta    = (G(i)-G(j))/qq;
            end
            alpha(i) = alpha(i) - delta;
            alpha(j) = alpha(j) + delta;
            s        = alpha(i) + alpha(j);
            if (s > C(i))
                if(alpha(i) > C(i))
                    alpha(i) = C(i);
                    alpha(j) = s - C(i);
                end
            else
                if(alpha(j) < 0)
                    alpha(j) = 0;
                    alpha(i) = s;
                end
            end
            if(s > C(j))
                if(alpha(j) > C(j))
                    alpha(j) = C(j);
                    alpha(i) = s - C(j);
                end
            else
                if(alpha(i) < 0)
                    alpha(i) = 0;
                    alpha(j) = s;
                end
            end
        end
        
        G = G + q(:,i)*(alpha(i) - oldAi) + q(:,j)*(alpha(j) - oldAj);
    end
    
	tsv = 0;
    for k = 1:np
        if ((alpha(k) > 0) && (alpha(k) < C(k)))
            tsv = tsv + 1;
        end
    end
    bias = 0.0;
    if (tsv > 0)
        for k = 1:np
            if ((alpha(k) > 0) && (alpha(k) < C(k)))
                bias = bias - y(k) * G(k);
            end
        end
        bias = bias / tsv;
    else
        bias = (G1 - G2) / 2.0;
    end
end