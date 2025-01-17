function [moy_estim, var_estim] = question3(signal, choix_estim)
[N,K] = size(signal); %signal est constitu� de K tranches de N �chantillons
res = zeros(2*N-1,K);
for k=1:K
    res(:,k) = question1(signal(:,k),N-1, 1, N, choix_estim);
end
moy_estim = mean(res,2);
var_estim = var(res,0,2);
end

    
