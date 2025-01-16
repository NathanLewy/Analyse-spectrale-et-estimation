% Calcule et affiche le périodogramme moyenné des signaux du DM1
close all
% signal1
L = 2048; % nombre total d'échantillons
% generation de signal1
[signal, DSP_th_signal1] = genere_signal1(L, 1);
% DSP_th_signal1 a 2048 échantillons
% dans l'intervalle des fréquences réduites [0, 1]
figure(1), hold on
N = 1024; tronq = 1;N1 = 1024;
while N>=64
    recouv = N/2;
    [DSP_estim, nb_tranches] = periodo_moyenne(signal, N, recouv, tronq);
    figure(1), plot((0:2*N-1)/2/N-1/2,10*log10([DSP_estim(N+1:end);DSP_estim(1:N)]));
    figure, hold on
    % calcule le périodogramme simple de signal à partir de l'échantillon
    % 1000
    perio_simple = abs(fft(signal(1000:1000+N-1),2*N)).^2/N;
    plot((0:2*N-1)/2/N-1/2,10*log10([perio_simple(N+1:end);perio_simple(1:N)]));
    plot((0:2*N-1)/2/N-1/2,10*log10([DSP_estim(N+1:end);DSP_estim(1:N)]));
    title(['Périodo simple et moyenné de signal1, tronq=',num2str(tronq),', N=',num2str(N),...
        ', nb\_tranches=',num2str(nb_tranches),', recouv=N/2']);
    plot((0:2*N1-1)/2/N1-1/2,10*log10([DSP_th_signal1(N1+1:end);DSP_th_signal1(1:N1)]));
    xlabel('fréquence réduite');
    ylabel('dB');
    legend({'simple','moyen','DSP\_th'},'Location','northeast');
    axis([-.5 .5 -60 10]);
    N = N/2;
end
figure(1), plot((0:2*N1-1)/2/N1-1/2,10*log10([DSP_th_signal1(N1+1:end);DSP_th_signal1(1:N1)]));
figure(1), xlabel('fréquence réduite');
figure(1), ylabel('dB');
figure(1), title(['Périodo moyennés de signal1 sur ',num2str(L),' échantillons, tronq=',...
    num2str(tronq),', recouv=N/2'])
figure(1), legend({'N=1024','N=512','N=256','N=128','N=64','DSP\_th'},'Location','northeast')
figure(1), axis([-.5 .5 -60 10])
