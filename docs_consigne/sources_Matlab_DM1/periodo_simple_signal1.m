% calcul du biais et de la variance du périodogramme simple appliqué à signal 1%
close all
N = 1024;
K = 32;
[signal1, DSP_th_signal1] = genere_signal1(N, K);
figure(1), hold on
figure(2), hold on
figure(3), hold on
figure(4), hold on
for k=1:5
    periodo_signal1 = abs(fft(signal1(1:N,:),2*N)).^2/N;
    figure(1), plot((0:N-1)/N,10*log10(periodo_signal1(1:2:end,1)))
    biais = mean(periodo_signal1(1:2:end,:),2)-DSP_th_signal1(1:2^k:end);
    figure(2), plot((0:N-1)/N,biais)
    figure(3), plot((0:N-1)/N,10*log10(var(periodo_signal1(1:2:end,:),0,2)))
    figure(4), plot((0:N-1)/N,10*log10(biais.^2+...
        var(periodo_signal1(1:2:end,:),0,2)))
    N = N/2;
end
N = 1024;
figure(1), plot((0:N-1)/N,10*log10(DSP_th_signal1(1:2:end)))
figure(1), axis([0 1 -60 10])
%figure(2), axis([0 1 -60 10])
figure(3), axis([0 1 -90 10])
figure(4), axis([0 1 -90 10])
figure(1), xlabel('fréquence réduite')
figure(2), xlabel('fréquence réduite')
figure(3), xlabel('fréquence réduite')
figure(4), xlabel('fréquence réduite')
figure(1), ylabel('dB')
%figure(2), ylabel('dB')
figure(3), ylabel('dB')
figure(4), ylabel('dB')
figure(1), title('périodogramme de signal1')
figure(2), title('biais du périodogramme de signal1')
figure(3), title('variance du périodogramme de signal1')
figure(4), title('erreur quadratique moyenne du périodogramme de signal1')
figure(1), legend({'N=1024','N=512','N=256','N=128','N=64','DSP\_th'},'Location','north')
figure(2), legend({'N=1024','N=512','N=256','N=128','N=64'},'Location','north')
figure(3), legend({'N=1024','N=512','N=256','N=128','N=64'},'Location','north')
figure(4), legend({'N=1024','N=512','N=256','N=128','N=64'},'Location','north')