import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signalmod
from scipy.interpolate import interp1d


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
bruit = np.random.randn(N) #bruit blanc

numerateur = [0.0154, 0.0461, 0.0461, 0.0154]  # Coefficients du numérateur (du terme de plus haut degré au terme constant)
denominateur = [1, -1.9903, 1.5717, -0.458]    # Coefficients du dénominateur (du terme de plus haut degré au terme constant)

# Définition du filtre avec les coefficients du numérateur et du dénominateur
systeme = signalmod.dlti(numerateur, denominateur)

# Calcul de la réponse impulsionnelle
temps, reponse_impulsionnelle = signalmod.dimpulse(systeme, n=110000)

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



def ex1_simple():

    #iterables
    liste_N = [128, 256, 512,1024]
    liste_col = ['red','blue','green','purple']
    signals = [signal1, bruit]
    signaltxt = ['signal1', 'bruit']
    modeles_theoriques = [pos_periodogramme_th*variance, np.ones(len(pos_periodogramme_th))]
    K=16

    for i in range(len(signals)):
        signal=signals[i]
        start_index=0
        plt.figure(figsize=(14, 8))
        plt.title("Périodogramme simple du " +signaltxt[i]+ " pour différentes valeurs de N")

        #itération pour différents N
        for n in liste_N:
            #réinit pour chaque N
            start_index = 0
            liste_periodo_simples=[]

            #calcul du périodograme pour K tranches
            for k in range(K):
                start_index += n
                tfd = np.fft.fft(signal[start_index:start_index+n])
                freqs= np.fft.fftfreq(n)
                periodogramme = np.abs(tfd) ** 2 / n

                # Ne garder que les fréquences positives
                pos_freqs = freqs[:len(freqs)//2]
                pos_periodogramme = periodogramme[:len(periodogramme)//2]

                #ajout aux listes
                liste_periodo_simples.append(pos_periodogramme)

            periodo_stats = compute_statistics(liste_periodo_simples)
            periodogramme_moyen = periodo_stats['mean']

            #pos_periodogramme_th
            #pos_freqs_th

            #creation des subplots
            plt.subplot(2,2,liste_N.index(n)+1)
            plt.grid(True)
            plt.xlabel("Fréquence")
            plt.ylabel("Amplitude en dB")
            #plt.yscale('log')

            #affichage dans la meme couleur d'une réalisation et de l'estimateur
            col_ = liste_col[liste_N.index(n)]
            plt.plot(pos_freqs, periodogramme_moyen ,label='périodogramme moyen sur K segments distincs pour N = ' + str(n),color=col_)
            plt.plot(pos_freqs, liste_periodo_simples[-1], label='périodogramme simple sur un segment', color=col_, alpha=0.5)
            plt.plot(pos_freqs_th, modeles_theoriques[i], label='périodogramme théorique', color='gray', alpha=0.8)

            #affichage de la ligne -60DB
            line_at_minus_60_dB = 0.001
            plt.axhline(y=line_at_minus_60_dB, color='black', linestyle='--', label='-60 dB')
            plt.legend()
            plt.yscale('log')
        plt.tight_layout()
        plt.show()







    
def ex2():
    #params
    N=7500
    f_ech = 1024 #Hz
    f_sinus = 140 #Hz
    A = np.sqrt(2)
    sigma = 0.08

    liste_t = np.array(range(N))/f_ech
    signal2 = np.sin(liste_t*f_sinus)*A + np.random.rand(N)*sigma

    Tranches = [64,128,256,512]
    liste_col = ['red','blue','green','purple']
    signals = [signal1, bruit, signal2]
    signalstxt = ['signal1', 'bruit', 'signal2']
    
    for i in range(len(signals)):
        signal=signals[i]
        plt.figure(figsize=(14, 8))
        plt.title("Periodogramme moyen avec différentes tranches de "+signalstxt[i])
        
        for taille_tranche in Tranches:
            periodogramme_liste=[]
            for i in range(1050//taille_tranche):
                start_index = i*taille_tranche
                tfd = np.fft.fft(signal[start_index:start_index+taille_tranche])
                freqs= np.fft.fftfreq(taille_tranche)
                periodogramme_liste.append(np.abs(tfd)* np.abs(tfd) / taille_tranche)
            periodo_moyen = compute_statistics(periodogramme_liste)['mean']

            # Separate and reorder the frequencies and corresponding amplitudes
            positive_freqs = freqs[freqs >= 0]
            negative_freqs = freqs[freqs < 0]

            positive_periodo_moyen = periodo_moyen[freqs >= 0]
            negative_periodo_moyen = periodo_moyen[freqs < 0]

            # Reassemble the lists: negative frequencies first, then positive
            freqs_reordered = np.concatenate((negative_freqs, positive_freqs))
            periodo_moyen_reordered = np.concatenate((negative_periodo_moyen, positive_periodo_moyen))

            plt.subplot(2, 2, Tranches.index(taille_tranche) + 1)
            plt.grid(True)
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude en dB")
            plt.yscale('log')

            # Plotting the reordered periodogram
            plt.plot(freqs_reordered, periodo_moyen_reordered, label='périodogramme moyen pour des tranches = ' + str(taille_tranche), color=liste_col[Tranches.index(taille_tranche)])

            #plot the comparative
            #plt.plot(freqs_reordered, periodo_simple, label='périodogramme simple pour des tranches = ' + str(taille_tranche), color=liste_col[Tranches.index(taille_tranche)], alpha = 0.5)
            #plt.plot(freqs_reordered, periodo_th, label='périodogramme theorique', color='gray')

            plt.legend()

        plt.tight_layout()
        plt.show()






if __name__ == "__main__":
    intro()
    ex1_simple()
    #print("valeurs theorique q1")
    #ex2()


