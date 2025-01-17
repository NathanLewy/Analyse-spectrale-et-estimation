function estim_autocor = question1(signal, M, i, N, choix_estim)
% retourne l'estimation de la fonction d'autocorrélation de signal(i:i+N-1)
% pour -M-1<k<M+1, l'estimateur est celui de Blackman-Tukey ou celui de
% Bartlett
signal = signal(:);
L = length(signal);
if i+N-1>L
    error(['Les arguments i=',num2str(i),' et N=',num2str(N), ...
        ' sont incompatibles avec la longueur de signal: ',num2str(L)]);
end
if M>=N
    error(['L''argument M=',num2str(M),' doit être < N=',num2str(N)]);
end
if strncmpi(choix_estim,'bla',3)% C'est l'estimateur de Blackman-Tukey
    estim_autocor = xcorr(signal(i:i+N-1),M,'unbiased');
elseif strncmpi(choix_estim,'bar',3)% C'est l'estimateur de Bartlett
    estim_autocor = xcorr(signal(i:i+N-1),M,'biased');
else
    error(['L''argument choix_estim = ', choix_estim,' doit commencer par Bla ou Bar']);
end
end