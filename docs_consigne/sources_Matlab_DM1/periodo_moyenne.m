function [DSP_estim, nb_tranches] = periodo_moyenne(signal, N, recouv, tronq)
% retourne le periodogramme moyenn� appliqu� � signal sur des transches
% de N �chantillons avec recouv �chantillons de recouvrement
% si la derni�re tranche est incompl�te, elle n'est pas prise en compte
% si tronq == 1 ou bien elle est prolong�e par des z�ros si tronq == 0
signal = signal(:); % transforme signal en une matrice colonne
L = length(signal);
if N>L && tronq==1
    error(['Le signal est trop court pour la valeur de N=',...
        num2str(N),' et l''option tronq==1']);
end
if recouv >= N || recouv<0
    error(['recouv=',num2str(recouv),' doit �tre >=0 et < N=',num2str(N)]);
end
if tronq~=0 && tronq~=1
    error('tronq doit prendre la valeur 0 ou 1');
end
DSP_estim = zeros(2*N,1); % La TFD sera calcul�e sur 2N points
debut_tranche_courante = 1;
nb_tranches = 0;
while debut_tranche_courante+N-1 <= L
    DSP_estim = DSP_estim + abs(fft(signal(debut_tranche_courante:...
        debut_tranche_courante+N-1),2*N)).^2/N;
    debut_tranche_courante = debut_tranche_courante+N-recouv;
    nb_tranches = nb_tranches+1;
end
if tronq==0 && debut_tranche_courante<=L %on tient compte de la derni�re tranche incompl�te
    DSP_estim = DSP_estim + abs(fft(signal(debut_tranche_courante:L),2*N)).^2/N;
    nb_tranches = nb_tranches+1;
end
 DSP_estim = DSP_estim/nb_tranches;
end