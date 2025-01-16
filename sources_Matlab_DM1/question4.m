function [DSP_estim, DSP_moy_estim, DSP_var_estim] = question4(signal,M)
% calcule la DSP avec la formule (4) (estimateur de Blackman-Tukey) de
% l'énoncé du DM
epsilon = 10^-12;
[N,K] = size(signal); %signal est constitué de K tranches de N échantillons
if M>=N
    error(['L''argument M=',num2str(M),' doit être < N=',num2str(N)]);
end
DSP_estim = zeros(2*M+1,K);
for k=1:K
    res_gamma = question1(signal(:,k),M, 1, N, 'Bla');
    % calcul de la TFD sur 2*M+1 points de res_gamma
    DSP_estim(:,k) = fft([res_gamma(M+1:end);res_gamma(1:M)]);
end
if max(max(abs(imag(DSP_estim)))) > epsilon
    error(['max(max(abs(imag(DSP_estim))))=',num2str(max(max(abs(imag(DSP_estim))))),...
        ', la TFD doit être réelle']);
end
DSP_estim = real(DSP_estim);
DSP_moy_estim = mean(DSP_estim,2);
DSP_var_estim = var(DSP_estim,0,2);
end