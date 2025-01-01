import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signalmod


def compute_statistics(estimations):
    return {"mean": np.mean(estimations, axis=0),
        "variance": np.var(estimations, axis=0),
        "std_dev": np.std(estimations, axis=0)}

N=2**17
variance = 0.5
sigma = np.sqrt(variance)

# Génération d'un signal pseudo-aléatoire gaussien
np.random.seed(42)
pre_signal = sigma * np.random.randn(N)  # Signal total

numerateur = [0.0154, 0.0461, 0.0461, 0.0154]  # Coefficients du numérateur (du terme de plus haut degré au terme constant)
denominateur = [1, -1.9903, 1.5717, -0.458]    # Coefficients du dénominateur (du terme de plus haut degré au terme constant)

# Définition du filtre avec les coefficients du numérateur et du dénominateur
systeme = signalmod.dlti(numerateur, denominateur)

# Calcul de la réponse impulsionnelle
temps, reponse_impulsionnelle = signalmod.dimpulse(systeme, n=10000)

# Convolution du signal d'entrée avec la réponse impulsionnelle pour obtenir la sortie
signal1 = np.convolve(pre_signal, reponse_impulsionnelle[0].flatten(), mode='full')  # mode 'full' pour inclure tous les points

# Tracer le signal d'entrée et la sortie filtrée
plt.figure()
plt.plot(pre_signal, label="signal d'entree")
plt.plot(signal1[:N], label="Signal de sortie après filtrage")  # Ne garder que les premiers N points pour la sortie
plt.title("Effet du filtre sur une réalisation du signal")
plt.xlabel("Temps (n)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()


def intro():
    # Tracer la réponse impulsionnelle
    plt.figure()
    plt.stem(temps[:50], reponse_impulsionnelle[0][:50])
    plt.title("Réponse impulsionnelle du filtre")
    plt.xlabel("Temps (n)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    
    #carré du module de la TFD du filtre
    tfd = np.fft.fft(reponse_impulsionnelle[0].flatten())
    periodogramme = np.abs(tfd)*np.abs(tfd)
    freqs = np.fft.fftfreq(len(periodogramme))
    
    
    # Ne garder que les fréquences positives
    global pos_periodogramme_th, pos_freqs_th
    pos_freqs_th = freqs[:len(freqs)//2]
    pos_periodogramme_th = periodogramme[:len(periodogramme)//2]

    # Tracer le periodogramme
    plt.figure()
    plt.plot(pos_freqs_th, pos_periodogramme_th, label='modèle de référence')

    # Ajouter la ligne constante à -60 dB (0.001 d'amplitude)
    line_at_minus_60_dB = 0.001
    plt.axhline(y=line_at_minus_60_dB, color='r', linestyle='--', label='-60 dB')

    plt.title("Périodogramme de la réponse impulsionnelle du filtre")
    plt.xlabel("Fréquence normalisée")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.yscale('log')  # Utilisation de l'échelle logarithmique sur l'axe des y
    plt.legend()
    plt.show()

def ex1():
    #signal1
    plt.figure(figsize=(14, 8))
    plt.title("Périodogramme simple du signal1 pour différentes valeurs de N")

    Nb_tranches = 250
    liste_statistiques = []
    liste_N = [128, 256, 512,1024]
    liste_col = ['red','blue','green','purple']
    #itération pour différents N
    for n in liste_N:
        liste_periodogramme = []

        #On moyenne sur plusieurs tranches dans 0,N de longueur n
        for tranche in range(Nb_tranches):
            start_index = np.random.randint(0,N-n)
            tfd = np.fft.fft(signal1[start_index:start_index+n])
            freqs= np.fft.fftfreq(n)
            periodogramme = np.abs(tfd)* np.abs(tfd) / n

            # Ne garder que les fréquences positives
            pos_freqs = freqs[:len(freqs)//2]
            pos_periodogramme = periodogramme[:len(periodogramme)//2]

            liste_periodogramme.append(pos_periodogramme)

        #calcul de l'estimateur du périodogramme
        liste_statistiques.append(compute_statistics(liste_periodogramme))

        #creation des subplots
        plt.subplot(2,2,liste_N.index(n)+1)
        plt.grid(True)
        plt.xlabel("Fréquence")
        plt.ylabel("Amplitude en dB")
        plt.yscale('log')

        #affichage dans la meme couleur d'une réalisation et de l'estimateur
        col_ = liste_col[liste_N.index(n)]
        plt.plot(pos_freqs,liste_statistiques[-1]['mean'], label='estimateur du périodogramme pour N = ' + str(n),color=col_)
        plt.plot(pos_freqs, pos_periodogramme, label='réalisation d un periodogramme pour N = '+str(n),color=col_, alpha=0.5)
        plt.plot(pos_freqs_th, pos_periodogramme_th, label='modèle de référence', color='black', alpha=0.6)

        #affichage de la ligne -60DB
        line_at_minus_60_dB = 0.001
        plt.axhline(y=line_at_minus_60_dB, color='black', linestyle='--', label='-60 dB')
        plt.legend()
    plt.tight_layout()
    plt.show()
    




if __name__ == "__main__":
    intro()
    ex1()


