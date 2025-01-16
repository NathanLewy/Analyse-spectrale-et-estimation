import numpy as np
import matplotlib.pyplot as plt

# Fonction pour générer signal1 et sa DSP théorique
def genere_signal1(N, K):
    np.random.seed(0)  # Fixer la graine pour la reproductibilité
    signal1 = np.random.randn(N, K)
    DSP_th_signal1 = np.ones(N)  # Exemple de DSP théorique simple
    return signal1, DSP_th_signal1

# Paramètres initiaux
N = 1024
K = 32
signal1, DSP_th_signal1 = genere_signal1(N, K)

# Création des figures
plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)

# Boucle principale
for k in range(1, 6):
    periodo_signal1 = np.abs(np.fft.fft(signal1[:N, :], 2 * N, axis=0))**2 / N
    plt.figure(1)
    plt.plot(np.arange(N) / N, 10 * np.log10(periodo_signal1[::2, 0]))
    
    biais = np.mean(periodo_signal1[::2, :], axis=1) - DSP_th_signal1[::2**k]
    plt.figure(2)
    plt.plot(np.arange(N) / N, biais)
    
    plt.figure(3)
    plt.plot(np.arange(N) / N, 10 * np.log10(np.var(periodo_signal1[::2, :], axis=1)))
    
    plt.figure(4)
    plt.plot(np.arange(N) / N, 10 * np.log10(biais**2 + np.var(periodo_signal1[::2, :], axis=1)))
    
    N = N // 2

# Plot des résultats finaux
N = 1024
plt.figure(1)
plt.plot(np.arange(N) / N, 10 * np.log10(DSP_th_signal1[::2]))
plt.axis([0, 1, -60, 10])

# Axes et labels pour toutes les figures
plt.figure(1)
plt.xlabel('Fréquence réduite')
plt.ylabel('dB')
plt.title('Périodogramme de signal1')
plt.legend(['N=1024', 'N=512', 'N=256', 'N=128', 'N=64', 'DSP_th'], loc='north')

plt.figure(2)
plt.xlabel('Fréquence réduite')
plt.ylabel('dB')
plt.title('Biais du périodogramme de signal1')
plt.legend(['N=1024', 'N=512', 'N=256', 'N=128', 'N=64'], loc='north')

plt.figure(3)
plt.xlabel('Fréquence réduite')
plt.ylabel('dB')
plt.title('Variance du périodogramme de signal1')
plt.legend(['N=1024', 'N=512', 'N=256', 'N=128', 'N=64'], loc='north')

plt.figure(4)
plt.xlabel('Fréquence réduite')
plt.ylabel('dB')
plt.title('Erreur quadratique moyenne du périodogramme de signal1')
plt.legend(['N=1024', 'N=512', 'N=256', 'N=128', 'N=64'], loc='north')

# Affichage des plots
plt.show()
