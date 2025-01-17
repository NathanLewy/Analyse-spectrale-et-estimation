function DSP_th_signal2 = modele_DSP_signal2(deb, N, Nfft)
% calcule la DSP théorique de signal2(i:i+N-1) avec une TFD sur Nfft points
if deb <=0
    error(['deb=',num2str(deb),' doit être > 0']);
end
if N <=0
    error(['N=',num2str(N),' doit être > 0']);
end
if deb+N-1 > 7500
    error(['Choix de i=',num2str(deb),' et N=',num2str(N),...
        ' incompatible avec la longueur du signal']);
end
L = 7500; % nombre d'échantillons
A1 = sqrt(2); % amplitude de la 1ère sinusoïde
A2 = sqrt(2)/100; % amplitude de la 2ème sinusoïde
Phi1 = 1.6367; % phase à l'origine de la 1ère sinusoïde
Phi2 = 1.0504; % phase à l'origine de la 2ème sinusoïde
f1 = 140; % fréquence en Hz de la 1ère sinusoïde
f2 = 180; % fréquence en Hz de la 2ème sinusoïde
fe = 1024; % fréquence d'échantillonnage en Hz
puissance_bb = (0.08)^2; % puissance du bruit blanc
indice_ech = (0:(L-1))';
signal2 = A1*sin(2*pi*f1/fe*indice_ech+Phi1)+A2*sin(2*pi*f2/fe*indice_ech+Phi2);
% calcul de la DSP théorique de la somme des deux sinusoïdes
nu = (0:(Nfft-1))'/Nfft;

DSP_th_signal2 = signal2(deb)*ones(Nfft,1);
for k=1:N-1
    DSP_th_signal2 = DSP_th_signal2 + signal2(deb+k)*exp(-2*1i*pi*k*nu);
end
DSP_th_signal2 = abs(DSP_th_signal2).^2/N + puissance_bb;
end