import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as signalmod
from scipy.interpolate import interp1d
from scipy.io import wavfile
from pydub import AudioSegment


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
    segment_length = 10000  # Longueur de chaque segment

    # Fonction pour calculer le périodogramme moyen via la TFD
    def compute_periodogram(signal, segment_length, window_func, K=None):
        # Découper le signal en segments
        num_segments = len(signal) // segment_length
        periodograms = []
        
        if K is not None:
            # Si K est spécifié, on prend K segments distincts
            for k in range(K):
                start_idx = k * segment_length
                end_idx = start_idx + segment_length
                if end_idx > len(signal):  # Vérifier que l'index de fin est dans les limites du signal
                    break
                segment = signal[start_idx:end_idx]
                
                # Appliquer la fenêtre
                windowed_segment = segment * window_func[:len(segment)]
                
                # Calculer la TFD
                spectrum = np.fft.fft(windowed_segment)
                
                # Calculer le périodogramme (module carré de la TFD)
                periodogram = np.abs(spectrum) ** 2 / segment_length
                periodograms.append(periodogram)
            
            # Moyenne des périodogrammes
            mean_periodogram = np.mean(periodograms, axis=0)
        else:
            # Si K n'est pas spécifié, on fait un découpage sans chevauchement
            for i in range(num_segments):
                start_idx = i * segment_length
                end_idx = start_idx + segment_length
                segment = signal[start_idx:end_idx]
                
                # Appliquer la fenêtre
                windowed_segment = segment * window_func[:len(segment)]
                
                # Calculer la TFD
                spectrum = np.fft.fft(windowed_segment)
                
                # Calculer le périodogramme (module carré de la TFD)
                periodogram = np.abs(spectrum) ** 2 / segment_length
                periodograms.append(periodogram[:len(periodogram) // 2])
            
            # Moyenne des périodogrammes
            mean_periodogram = np.mean(periodograms, axis=0)
        
        # Fréquences associées à la TFD
        freqs = np.fft.fftfreq(segment_length, d=1/sample_rate)[:len(periodogram) // 2]
        
        return freqs[:segment_length // 2], mean_periodogram[:segment_length // 2]
    # Fenêtres à utiliser : rectangulaire, Hamming et Hanning
    windows = {
        'Rectangulaire': np.ones(segment_length),
        'Hamming': np.hamming(segment_length),
        'Hanning': np.hanning(segment_length)
    }
    
    # Affichage des périodogrammes moyens pour chaque fenêtre
    plt.figure(figsize=(15, 10))
    plt.title(f"Périodogramme moyen avec différentes fenêtres")
    liste_col = ['red', 'blue', 'green', 'purple']
    for i, (window_name, window_func) in enumerate(windows.items()):
        f, P = compute_periodogram(audio_data, segment_length, window_func)
        f, Pk = compute_periodogram(audio_data, segment_length, window_func, 32)
        plt.plot(f, P, color=liste_col[i], label=window_name+"avec chevauchement", alpha=0.5)
        plt.plot(f, Pk, color=liste_col[i+1], label=window_name+"avec K segments distincts")
        plt.xlabel("Fréquence")
        plt.ylabel("Densité de puissance")
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
    plt.show()

ex4()