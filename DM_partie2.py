import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signalmod
from scipy.interpolate import interp1d
from scipy.io import wavfile
from pydub import AudioSegment

def compute_statistics(estimations):
    return {"mean": np.mean(estimations, axis=0),
        "variance": np.var(estimations, axis=0),
        "std_dev": np.std(estimations, axis=0)}

def variance_periodogramme_bb(N, nu_p, sigma):
    #calcule la variance théorique de l'estimateur du periodogramme pour un bruit blanc
    nu_p = np.pi * nu_p
    terme = (np.sin(2 * N * nu_p) / (N * np.sin(2 * nu_p))) ** 2
    variance = sigma**4 * (1 + terme)
    return variance

N=2**17
variance = 0.5
sigma = np.sqrt(variance)


# Génération d'un signal pseudo-aléatoire gaussien
np.random.seed(42)
pre_signal = sigma * np.random.randn(N)  # Signal total

sigma_bb= np.sqrt(2)
bruit = sigma_bb * np.random.randn(N) #bruit blanc

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



def ex1_signal_et_bruit():
    # Iterables
    liste_N = [128, 256, 512, 1024]
    liste_col = ['red', 'blue', 'green', 'purple']
    signals = [signal1, bruit]
    signaltxt = ['signal1', 'bruit']
    modeles_theoriques = [pos_periodogramme_th * variance, (sigma_bb**2) * np.ones(len(pos_periodogramme_th))]
    K = 16

    for i in range(len(signals)):
        signal = signals[i]

        # Figure pour les périodogrammes
        plt.figure(figsize=(14, 8))
        plt.suptitle("Périodogramme simple du " + signaltxt[i] + " pour différentes valeurs de N et K=" + str(K))

        # Itération pour différents N
        for n in liste_N:
            start_index = 0
            liste_periodo_simples = []

            # Calcul du périodograme pour K tranches
            for k in range(K):
                start_index += n
                tfd = np.fft.fft(signal[start_index:start_index + n])
                freqs = np.fft.fftfreq(n)
                periodogramme = np.abs(tfd) ** 2 / n

                # Ne garder que les fréquences positives
                pos_freqs = freqs[:len(freqs) // 2]
                pos_periodogramme = periodogramme[:len(periodogramme) // 2]

                # Ajout aux listes
                liste_periodo_simples.append(pos_periodogramme)

            periodo_stats = compute_statistics(liste_periodo_simples)
            periodogramme_moyen = periodo_stats['mean']

            # Création des subplots pour les périodogrammes
            plt.subplot(2, 2, liste_N.index(n) + 1)
            plt.grid(True)
            plt.xlabel("Fréquence")
            plt.ylabel("Amplitude en dB")

            # Affichage dans la même couleur d'une réalisation et de l'estimateur
            col_ = liste_col[liste_N.index(n)]
            plt.plot(pos_freqs, periodogramme_moyen, label='périodogramme moyen sur K segments pour N = ' + str(n), color=col_)
            plt.plot(pos_freqs, liste_periodo_simples[-1], label='périodogramme simple sur un segment', color=col_, alpha=0.5)
            plt.plot(pos_freqs_th, modeles_theoriques[i], label='périodogramme théorique', color='gray', alpha=0.8)

            # Affichage de la ligne -60DB
            plt.axhline(y=0.001, color='black', linestyle='--', label='-60 dB')
            plt.legend()
            plt.yscale('log')

        plt.tight_layout()
        plt.show()

        # Figure pour biais et variance
        plt.figure(figsize=(14, 8))
        plt.suptitle("Biais et variance du périodogramme simple du " + signaltxt[i] + " pour différentes valeurs de N et K=" + str(K))

        for n in liste_N:
            start_index = 0
            liste_periodo_simples = []

            for k in range(K):
                start_index += n
                tfd = np.fft.fft(signal[start_index:start_index + n])
                freqs = np.fft.fftfreq(n)
                periodogramme = np.abs(tfd) ** 2 / n

                pos_freqs = freqs[:len(freqs) // 2]
                pos_periodogramme = periodogramme[:len(periodogramme) // 2]
                liste_periodo_simples.append(pos_periodogramme)

            periodo_stats = compute_statistics(liste_periodo_simples)
            periodogramme_moyen = periodo_stats['mean']

            # Interpoler le modèle théorique sur les fréquences pos_freqs
            interpolateur = interp1d(pos_freqs_th, modeles_theoriques[i], bounds_error=False, fill_value="extrapolate")
            model_theorique_interp = interpolateur(pos_freqs)
            biais_periodogramme = periodogramme_moyen - model_theorique_interp
            variance_periodogramme = periodo_stats['variance']

            # Création des subplots pour biais et variance
            plt.subplot(2, 2, liste_N.index(n) + 1)
            plt.grid(True)
            plt.xlabel("Fréquence")
            plt.ylabel("Amplitude en dB")
            plt.plot(pos_freqs, biais_periodogramme, label='biais du périodogramme moyen pour N = ' + str(n), color=col_, alpha=0.5)
            plt.plot(pos_freqs, variance_periodogramme, label='variance du périodogramme pour N = ' + str(n), color=col_)

            #affiche le résultat théorique pour le bruit blanc de v
            if signaltxt[i]=='bruit':
                plt.plot(pos_freqs, variance_periodogramme_bb(n,pos_freqs, sigma_bb), label='variance théorique', color='gray')
            plt.legend()
            plt.yscale('log')

            # Affichage de la ligne -60DB
            plt.axhline(y=0.001, color='black', linestyle='--', label='-60 dB')
            plt.legend()
            plt.yscale('log')

        plt.tight_layout()
        plt.show()

        



def ex1_sinus():
    
    K = 32  # Nombre de tranches pour l'estimation
    liste_N = [128, 256, 512, 1024]  # Différentes valeurs de N
    liste_col = ['red', 'blue', 'green', 'purple']  # Couleurs pour les graphes
    
    # Figure pour les périodogrammes
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"DSP d'un sinus pour différentes valeurs de N et K={K}")

    # Itération pour différentes valeurs de N
    for i, N in enumerate(liste_N):
        # Génération du signal sinusoïdal
        A = 1.0  # Amplitude du signal
        nu_0 = 10/N # Fréquence réduite (par exemple, ν₀ = 0.1)
        phi = 0.2  # Phase initiale
        t = np.arange(0, K * N)  # Temps discret
        signal = A * np.sin(2 * np.pi * nu_0 * t + phi)  # Signal sinusoïdal

        # Initialisation des listes pour le périodogramme
        liste_periodo_simples = []

        # Découpage en K segments et calcul des périodogrammes
        for k in range(K):
            start_index = k * N
            segment = signal[start_index:start_index + N]
            tfd = np.fft.fft(segment)  # Transformée de Fourier discrète
            freqs = np.fft.fftfreq(N)  # Fréquences associées
            periodogramme = np.abs(tfd) ** 2 / N  # Périodogramme

            # Ne garder que les fréquences positives
            pos_freqs = freqs[:N // 2]
            pos_periodogramme = periodogramme[:N // 2]
            liste_periodo_simples.append(pos_periodogramme)

        # Moyenne des périodogrammes sur K segments
        periodo_stats = compute_statistics(liste_periodo_simples)
        periodogramme_moyen = periodo_stats['mean']

        model_theorique = N * A**2 / 4

        # Création des subplots pour chaque N
        plt.subplot(2, 2, i + 1)
        plt.grid(True)
        plt.xlabel("Fréquence réduite")
        plt.ylabel("Amplitude")
        plt.title(f"Différents tracés DSP d'un signal sinusoidal N = {N}")
        col_ = liste_col[i]

        # Tracé du périodogramme moyen, simple et théorique
        plt.plot(pos_freqs, periodogramme_moyen, label="Périodogramme moyen sur K tranches de taile N", color=col_)
        plt.plot(pos_freqs, liste_periodo_simples[-1], label="Périodogramme simple pour une tranche de taille N", color=col_, alpha=0.5)
        plt.stem([nu_0], [model_theorique], linefmt='black', label="Périodogramme théorique", basefmt=" ")

        plt.legend()
        plt.yscale('log')

    plt.tight_layout()
    plt.show()


    K = 32  # Nombre de tranches pour l'estimation
    liste_N = [128, 256, 512, 1024]  # Différentes valeurs de N
    liste_col = ['red', 'blue', 'green', 'purple']  # Couleurs pour les graphes
    
    # Figure pour les périodogrammes
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"DSP d'un sinus pour différentes valeurs de N et K={K}")

    # Itération pour différentes valeurs de N
    for i, N in enumerate(liste_N):
        # Génération du signal sinusoïdal
        A = 1.0  # Amplitude du signal
        nu_0 = (10 + 1/2)/N # Fréquence réduite (par exemple, ν₀ = 0.1)
        phi = 0.2  # Phase initiale
        t = np.arange(0, K * N)  # Temps discret
        signal = A * np.sin(2 * np.pi * nu_0 * t + phi)  # Signal sinusoïdal

        # Initialisation des listes pour le périodogramme
        liste_periodo_simples = []

        # Découpage en K segments et calcul des périodogrammes
        for k in range(K):
            start_index = k * N
            segment = signal[start_index:start_index + N]
            tfd = np.fft.fft(segment)  # Transformée de Fourier discrète
            freqs = np.fft.fftfreq(N)  # Fréquences associées
            periodogramme = np.abs(tfd) ** 2 / N  # Périodogramme

            # Ne garder que les fréquences positives
            pos_freqs = freqs[:N // 2]
            pos_periodogramme = periodogramme[:N // 2]
            liste_periodo_simples.append(pos_periodogramme)

        # Moyenne des périodogrammes sur K segments
        periodo_stats = compute_statistics(liste_periodo_simples)
        periodogramme_moyen = periodo_stats['mean']
        model_theorique = N * A**2 / 4

        # Création des subplots pour chaque N
        plt.subplot(2, 2, i + 1)
        plt.grid(True)
        plt.xlabel("Fréquence réduite")
        plt.ylabel("Amplitude")
        plt.title(f"Différents tracés DSP d'un signal sinusoidal N = {N}")
        col_ = liste_col[i]

        # Tracé du périodogramme moyen, simple et théorique
        plt.plot(pos_freqs, periodogramme_moyen, label="Périodogramme moyen sur K tranches de taile N", color=col_)
        plt.plot(pos_freqs, liste_periodo_simples[-1], label="Périodogramme simple pour une tranche de taille N", color=col_, alpha=0.4)
        plt.stem([nu_0], [model_theorique], linefmt='black', label="Périodogramme théorique", basefmt=" ")

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
    modeles_theoriques = [pos_periodogramme_th * variance, (sigma_bb**2) * np.ones(len(pos_periodogramme_th)),np.array([N * A**2 / 4])]
    modeles_theoriques_frequences = [pos_freqs_th, pos_freqs_th, [f_sinus/(f_ech*2*np.pi)]]
    
    for i in range(len(signals)):
        signal=signals[i]
        plt.figure(figsize=(14, 8))
        plt.title("Periodogramme moyen avec différentes tranches de "+signalstxt[i])
        
        for taille_tranche in Tranches:
            periodogramme_liste=[]
            periodogramme_liste_chevauchement = []
            for j in range(1050//taille_tranche):
                start_index = j*taille_tranche
                tfd = np.fft.fft(signal[start_index:start_index+taille_tranche])
                freqs= np.fft.fftfreq(taille_tranche)
                periodogramme_liste.append(np.abs(tfd)* np.abs(tfd) / taille_tranche)

            for j in range(1050-taille_tranche):
                start_index = j
                tfd = np.fft.fft(signal[start_index:start_index+taille_tranche])
                freqs= np.fft.fftfreq(taille_tranche)
                periodogramme_liste_chevauchement.append(np.abs(tfd)* np.abs(tfd) / taille_tranche)

            periodo_moyen = compute_statistics(periodogramme_liste)['mean']
            periodo_moyen_chevauchement = compute_statistics(periodogramme_liste_chevauchement)['mean']

            # Separate and reorder the frequencies and corresponding amplitudes
            positive_freqs = freqs[freqs >= 0]
            positive_periodo_moyen = periodo_moyen[freqs >= 0]
            positive_periodo_moyen_chevauchement = periodo_moyen_chevauchement[freqs >= 0]


            #plt.subplot(2, 2, Tranches.index(taille_tranche) + 1)
            plt.grid(True)
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude en dB")
            plt.yscale('log')

            # Plotting the reordered periodogram
            plt.plot(positive_freqs, positive_periodo_moyen, label='périodogramme moyen pour des tranches = ' + str(taille_tranche), color=liste_col[Tranches.index(taille_tranche)], alpha=0.4)
            plt.plot(positive_freqs, positive_periodo_moyen_chevauchement, label='périodogramme moyen avec chevauchement pour des tranches = ' + str(taille_tranche), color=liste_col[Tranches.index(taille_tranche)])
        
        if i==2:
            plt.stem(modeles_theoriques_frequences[i], modeles_theoriques[i], linefmt='black', label="Périodogramme théorique", basefmt=" ")
        else:
            plt.plot(modeles_theoriques_frequences[i], modeles_theoriques[i], label='périodogramme théorique', color='black', alpha=0.8)

        # Affichage de la ligne -60DB
        plt.axhline(y=0.001, color='black', linestyle='--', label='-60 dB')

        plt.legend()

        plt.tight_layout()
        plt.show()


def ex3():
    #params
    N=7500
    f_ech = 1024 #Hz
    f_sinus = 140 #Hz
    A = np.sqrt(2)
    sigma = 0.08

    liste_t = np.array(range(N))/f_ech
    signal2 = np.sin(liste_t*f_sinus)*A + np.random.rand(N)*sigma

    taille_tranche = 512
    liste_col = ['red','blue','green','purple']
    signals = [signal1, bruit, signal2]
    signalstxt = ['signal1', 'bruit', 'signal2']
    fenetres = [np.ones(taille_tranche), np.hamming(taille_tranche), np.bartlett(taille_tranche), np.blackman(taille_tranche)]
    fenetretxt = ['rectangle', 'hamming', 'triangulaire', 'blackman']
    modeles_theoriques = [pos_periodogramme_th * variance, (sigma_bb**2) * np.ones(len(pos_periodogramme_th)),np.array([N * A**2 / 4])]
    modeles_theoriques_frequences = [pos_freqs_th, pos_freqs_th, [f_sinus/(f_ech*2*np.pi)]]
    
    for i in range(len(signals)):
        signal=signals[i]
        plt.figure(figsize=(14, 8))
        plt.title(f"Periodogramme moyen de {signalstxt[i]} avec différentes fenêtres")
        
        for f in range(len(fenetres)):
            fenetre = fenetres[f]
            periodogramme_liste=[]
            periodogramme_liste_chevauchement = []
            for j in range(1050//taille_tranche):
                start_index = j*taille_tranche
                tfd = np.fft.fft(signal[start_index:start_index+taille_tranche]*fenetre)
                freqs= np.fft.fftfreq(taille_tranche)
                periodogramme_liste.append(np.abs(tfd)* np.abs(tfd) / taille_tranche)

            for j in range(1050-taille_tranche):
                start_index = j
                tfd = np.fft.fft(signal[start_index:start_index+taille_tranche]*fenetre)
                freqs= np.fft.fftfreq(taille_tranche)
                periodogramme_liste_chevauchement.append(np.abs(tfd)* np.abs(tfd) / taille_tranche)

            periodo_moyen = compute_statistics(periodogramme_liste)['mean']
            periodo_moyen_chevauchement = compute_statistics(periodogramme_liste_chevauchement)['mean']

            # Separate and reorder the frequencies and corresponding amplitudes
            positive_freqs = freqs[freqs >= 0]
            positive_periodo_moyen = periodo_moyen[freqs >= 0]
            positive_periodo_moyen_chevauchement = periodo_moyen_chevauchement[freqs >= 0]


            #plt.subplot(2, 2, Tranches.index(taille_tranche) + 1)
            plt.grid(True)
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude en dB")
            plt.yscale('log')

            # Plotting the reordered periodogram
            plt.plot(positive_freqs, positive_periodo_moyen, label='périodogramme moyen pour une fenêtre de = ' + str(fenetretxt[f]), color=liste_col[f], alpha=0.4)
            plt.plot(positive_freqs, positive_periodo_moyen_chevauchement, label='périodogramme moyen avec chevauchement pour une fenêtre de = ' + str(fenetretxt[f]), color=liste_col[f])
        
        if i==2:
            plt.stem(modeles_theoriques_frequences[i], modeles_theoriques[i], linefmt='black', label="Périodogramme théorique", basefmt=" ")
        else:
            plt.plot(modeles_theoriques_frequences[i], modeles_theoriques[i], label='périodogramme théorique', color='black', alpha=0.8)

        # Affichage de la ligne -60DB
        plt.axhline(y=0.001, color='black', linestyle='--', label='-60 dB')

        plt.legend()

        plt.tight_layout()
        plt.show()




def ex4():
    # Spécifier le chemin relatif du fichier dans le sous-dossier
    file_path = 'docs_consigne/signal'
    
    # Charger les données audio avec np.fromfile
    try:
        with open(file_path, 'rb') as f:
            # Lire les données brutes du fichier audio
            audio_data = np.fromfile(f, dtype=np.int16)  # Assurez-vous que le type est correct pour votre fichier
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {e}")
        return

    # Afficher la longueur du signal et la fréquence d'échantillonnage
    sample_rate = 44100  # Exemple de fréquence d'échantillonnage, à ajuster selon le fichier
    print(f"Fréquence d'échantillonnage : {sample_rate} Hz")
    print(f"Longueur du signal : {len(audio_data)} échantillons")
    
    # Définir le nombre de segments pour le périodogramme moyen
    segment_length = 1024  # Longueur de chaque segment
    overlap = 512  # Chevauchement entre les segments
    noverlap = segment_length - overlap

    # Fonction pour calculer le périodogramme moyen
    def compute_periodogram(signal, segment_length, noverlap, window_func):
        # Découper le signal en segments
        f, Pxx = plt.psd(signal, NFFT=segment_length, Fs=sample_rate, noverlap=noverlap, window=window_func)
        return f, Pxx

    # Fenêtres à utiliser : rectangulaire, Hamming et Hanning
    windows = {
        'Rectangulaire': np.ones(segment_length),
        'Hamming': np.hamming(segment_length),
        'Hanning': np.hanning(segment_length)
    }
    
    # Affichage des périodogrammes moyens pour chaque fenêtre
    plt.figure(figsize=(15, 10))

    for i, (window_name, window_func) in enumerate(windows.items()):
        plt.subplot(3, 1, i + 1)
        f, Pxx = compute_periodogram(audio_data, segment_length, noverlap, window_func)
        plt.semilogy(f, Pxx)
        plt.title(f"Périodogramme moyen avec fenêtre {window_name}")
        plt.xlabel("Fréquence [Hz]")
        plt.ylabel("Densité de puissance [V^2/Hz]")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ex4()
    intro()
    ex1_signal_et_bruit()
    ex1_sinus()
    ex2()
    ex3()


