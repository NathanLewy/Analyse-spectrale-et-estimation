function signal2 = genere_signal2()
% gen�re le signal2 d�crit dans l'�nonc� du DM1, deuxi�me partie, question
% 2
N = 7500; % nombre d'�chantillons
A1 = sqrt(2); % amplitude de la 1�re sinuso�de
A2 = sqrt(2)/100; % amplitude de la 2�me sinuso�de
Phi1 = 1.6367; % phase � l'origine de la 1�re sinuso�de
Phi2 = 1.0504; % phase � l'origine de la 2�me sinuso�de
f1 = 140; % fr�quence en Hz de la 1�re sinuso�de
f2 = 180; % fr�quence en Hz de la 2�me sinuso�de
fe = 1024; % fr�quence d'�chantillonnage en Hz
puissance_bb = (0.08)^2; % puissance du bruit blanc
indice_ech = (0:(N-1))';
signal2 = A1*sin(2*pi*f1/fe*indice_ech+Phi1)+A2*sin(2*pi*f2/fe*indice_ech+Phi2);
signal2 = signal2 + sqrt(puissance_bb)*randn(N,1);
end