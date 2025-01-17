function [signal1, DSP_th_signal1] = genere_signal1(N, K)
% génère K réalisations de N échantillons du signal1 du DM (deuxième partie)
num = [.0154 .0461 .0461 .0154];
den = [1 -1.9903 1.5717 -.458];
variance_bb = 0.5;
signal1 = filter(num,den, sqrt(variance_bb)*randn(N*K,1));
signal1 = reshape(signal1,N,K);
% Calcul de la DSP théorique de signal1 sur 2048 points
% Il sera ainsi possible de sous-échantillonner la DSP théorique
% pour l'obtenir sur un nombre de points = une puissance de 2 inférieure
% ou égale à 2048
rep_imp = filter(num,den,[1;zeros(2047,1)]);
DSP_th_signal1 = variance_bb*abs(fft(rep_imp)).^2;
end
