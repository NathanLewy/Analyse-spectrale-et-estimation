% Calcule et affiche le p�riodogramme moyenn� des signaux du DM1
close all
% signal1
L = 1050; % nombre total d'�chantillons
fe = 1024; %fr�quence d'�chantillonnage
% generation de signal2
signal = genere_signal2();
deb = 1; %indice du 1er �chantillon de signal pris en compte
signal = signal(deb:deb+L-1);
% DSP_th_signal1 a 2048 �chantillons
% dans l'intervalle des fr�quences r�duites [0, 1]
figure(1), hold on
N = 1024; tronq = 1;N1 = 1024;
while N>=64
    recouv = N/2;
    [DSP_estim, nb_tranches] = periodo_moyenne(signal, N, recouv, tronq);
    DSP_th_signal = modele_DSP_signal2(deb, N, 2*N1);
    figure(1), plot(((0:2*N-1)/2/N-1/2)*fe,10*log10([DSP_estim(N+1:end);DSP_estim(1:N)]));
    figure, hold on
    % calcule le p�riodogramme simple de signal � partir de l'�chantillon
    % deb
    perio_simple = abs(fft(signal(1:N),2*N)).^2/N;
    plot(((0:2*N-1)/2/N-1/2)*fe,10*log10([perio_simple(N+1:end);perio_simple(1:N)]));
    plot(((0:2*N-1)/2/N-1/2)*fe,10*log10([DSP_estim(N+1:end);DSP_estim(1:N)]));
    title(['P�riodo simple et moyenn� de signal2, tronq=',num2str(tronq),', N=',num2str(N),...
        ', nb\_tranches=',num2str(nb_tranches),', recouv=N/2']);
    plot(((0:2*N1-1)/2/N1-1/2)*fe,10*log10([DSP_th_signal(N1+1:end);DSP_th_signal(1:N1)]));
    xlabel('fr�quence r�duite');
    ylabel('dB');
    legend({'simple','moyen','DSP\_th'},'Location','northeast');
    axis([-512 512 -25 30])
    N = N/2;
end
figure(1), xlabel('fr�quence r�duite');
figure(1), ylabel('dB');
figure(1), title(['P�riodo moyenn�s de signal2 sur ',num2str(L),' �chantillons, tronq=',...
    num2str(tronq),', recouv=N/2'])
figure(1), legend({'N=1024','N=512','N=256','N=128','N=64'},'Location','northeast')
figure(1), axis([-512 512 -25 30])
