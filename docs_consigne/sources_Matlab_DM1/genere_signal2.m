function signal2 = genere_signal2()
% genère le signal2 décrit dans l'énoncé du DM1, deuxième partie, question
% 2
N = 7500; % nombre d'échantillons
A1 = sqrt(2); % amplitude de la 1ère sinusoïde
A2 = sqrt(2)/100; % amplitude de la 2ème sinusoïde
Phi1 = 1.6367; % phase à l'origine de la 1ère sinusoïde
Phi2 = 1.0504; % phase à l'origine de la 2ème sinusoïde
f1 = 140; % fréquence en Hz de la 1ère sinusoïde
f2 = 180; % fréquence en Hz de la 2ème sinusoïde
fe = 1024; % fréquence d'échantillonnage en Hz
puissance_bb = (0.08)^2; % puissance du bruit blanc
indice_ech = (0:(N-1))';
signal2 = A1*sin(2*pi*f1/fe*indice_ech+Phi1)+A2*sin(2*pi*f2/fe*indice_ech+Phi2);
signal2 = signal2 + sqrt(puissance_bb)*randn(N,1);
end