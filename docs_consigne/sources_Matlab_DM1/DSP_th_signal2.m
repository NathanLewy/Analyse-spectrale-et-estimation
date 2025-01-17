% calcul des DSP th�oriques de signal2 pour des observations de N
% �chantillons
close all
Nfft = 2^16;% doit �tre une puissance de 2
deb = 1;
N = 1024; % doit �tre une puissance de 2
fe = 1024; % fr�quence d'�chantillonnage en Hz
debut_zoom_freq = 120; % fr�quence basse du zoom en Hz
indice_deb_zoom = floor(debut_zoom_freq*Nfft/fe);
fin_zoom_freq = 200; % frequence haute du zoom en Hz
indice_fin_zoom = ceil(fin_zoom_freq*Nfft/fe);
frequences = (0:Nfft-1)/Nfft*fe;
zoom_frequences = frequences(indice_deb_zoom:ceil(fin_zoom_freq*Nfft/fe));
figure(1), hold on
figure(2), hold on
for k=1:5
    DSP_theo_signal2 = modele_DSP_signal2(deb, N, Nfft);
    figure(1), plot(frequences-fe/2,10*log10([DSP_theo_signal2(Nfft/2+1:end);DSP_theo_signal2(1:Nfft/2)]))
    figure(2), plot(zoom_frequences,10*log10(DSP_theo_signal2(indice_deb_zoom:indice_fin_zoom)))
    N = N/2;
end
figure(1), xlabel('fr�quence en Hz');
figure(1), ylabel('dB');
figure(1), title('DSP th�orique de signal2 observ� sur N �chantillons')
figure(1), legend({'N=1024','N=512','N=256','N=128','N=64'},'Location','north')
figure(1), axis([-512 512 -25 30])
figure(2), xlabel('fr�quence en Hz');
figure(2), ylabel('dB');
figure(2), title('zoom DSP th�orique de signal2 observ� sur N �chantillons')
figure(2), legend({'N=1024','N=512','N=256','N=128','N=64'},'Location','northeast')
figure(2), axis([120 200 -25 30])
