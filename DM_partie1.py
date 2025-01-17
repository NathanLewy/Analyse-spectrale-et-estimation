import numpy as np
import matplotlib.pyplot as plt


def blackman_tukey_estimator(signal, N, M):
    """Calcule les coefficients de corrélation selon l'estimateur de Blackman-Tukey."""
    L = len(signal)
    
    gamma_x = []
    for k in range(-M, M + 1):
        autocov = 0
        for n in range(N - abs(k)):
            if 0 <= n < L and 0 <= n + np.abs(k) < L:
                autocov += signal[n] * signal[n + np.abs(k)]
        gamma_x.append(autocov / (N - np.abs(k)))
    
    return np.array(gamma_x)

def bartlett_estimator(signal, N, M):
    """Calcule les coefficients de corrélation selon l'estimateur de Bartlett."""
    L = len(signal)
    
    gamma_x = []
    for k in range(-M, M + 1):
        autocov = 0
        for n in range(N - np.abs(k)):
            if 0 <= n < L and 0 <= n + k < L:
                autocov += signal[n] * signal[n + np.abs(k)]
        gamma_x.append(autocov / N)
    
    return np.array(gamma_x)


def compute_statistics(estimations):
    return {"mean": np.mean(estimations, axis=0),
        "variance": np.var(estimations, axis=0),
        "std_dev": np.std(estimations, axis=0)}

def ex1():
    # Paramètres d'entrée
    N = 1024  # Exemple: nombre d'échantillons
    M = 1023   # Exemple: horizon de calcul


    # Génération d'un signal aléatoire pour l'exemple
    np.random.seed(42)  # Pour des résultats reproductibles
    signal = np.random.randn(N)  # Signal aléatoire

    # Extraction des échantillons à partir de i
    signal = signal[:N]
    # Calcul des coefficients selon les deux méthodes
    gamma_bt = blackman_tukey_estimator(signal, N, M)
    gamma_bartlett = bartlett_estimator(signal, N, M)

    # Visualisation des coefficients
    k_values = np.arange(-M, M + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, gamma_bt, color="blue", label="Blackman-Tukey")
    plt.plot(k_values, gamma_bartlett, color="orange", label="Bartlett")
    plt.title("Estimation des coefficients de corrélation")
    plt.xlabel("Indice k")
    plt.ylabel("Coefficient de corrélation")
    plt.legend()
    plt.grid()
    plt.show()




def ex2():
    # Paramètres d'entrée
    N = 1024  # Exemple: nombre d'échantillons
    i = 0    # Exemple: indice du premier échantillon
    M = 1023   # Exemple: horizon de calcul
    variance = 0.1
    sigma = np.sqrt(variance)

    # Vérification des valeurs
    if i < 0 or N <= 0 or M <= 0 or M>=N:
        print("Erreur: N, i, et M doivent être positifs et valides.")
        return

    # Génération d'un signal aléatoire pour l'exemple
    np.random.seed(47)  # Pour des résultats reproductibles
    signal = sigma * np.random.randn(i + N)  # Signal aléatoire

    # Extraction des échantillons à partir de i
    signal = signal[i:i + N]
    # Calcul des coefficients selon les deux méthodes
    gamma_bt = blackman_tukey_estimator(signal, N, M)
    gamma_bartlett = bartlett_estimator(signal, N, M)
    gamma_th = np.zeros(2*M+1)
    gamma_th[M]=variance

    # Visualisation des coefficients
    k_values = np.arange(-M, M + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, gamma_bt, color="blue", label="Blackman-Tukey")
    plt.plot(k_values, gamma_bartlett, color="orange", label="Bartlett")
    plt.plot(k_values, gamma_th, label='Gamma théorique', color="red", alpha=0.5)
    plt.title("Estimation des coefficients de corrélation")
    plt.xlabel("Indice k")
    plt.ylabel("Coefficient de corrélation")
    plt.legend()
    plt.grid()

    # Affichage des valeurs en k=0
    value_bt = gamma_bt[M]  # Index correspondant à k=0
    value_bartlett = gamma_bartlett[M]
    plt.annotate(f"BT: {value_bt:.3f}", xy=(0, value_bt), xytext=(50, 10),
                 textcoords="offset points", arrowprops=dict(arrowstyle="->"), ha='center')
    plt.annotate(f"Bartlett: {value_bartlett:.3f}", xy=(0, value_bartlett), xytext=(-50, -30),
                 textcoords="offset points", arrowprops=dict(arrowstyle="->"), ha='center')

    print(f"Valeur Blackman-Tukey en k=0: {value_bt}")
    print(f"Valeur Bartlett en k=0: {value_bartlett}")

    plt.show()



def ex3():
    # Paramètres d'entrée
    N = 1024  # Nombre d'échantillons par tranche
    M = N-1   # Horizon de calcul
    K = 16    # Nombre de tranches
    variance = 0.1
    sigma = np.sqrt(variance)
    

    # Génération d'un signal pseudo-aléatoire gaussien
    np.random.seed(42)
    total_signal = sigma * np.random.randn(K * N)  # Signal total (K tranches)

    # Stocker les estimations
    gamma_bt_list = []
    gamma_bartlett_list = []

    for k in range(K):
        # Extraire la tranche k
        signal = total_signal[k * N:(k + 1) * N]
        
        # Calculer les estimations
        gamma_bt = blackman_tukey_estimator(signal, N, M)
        gamma_bartlett = bartlett_estimator(signal, N, M)
        
        gamma_bt_list.append(gamma_bt)
        gamma_bartlett_list.append(gamma_bartlett)

    # Convertir en tableaux NumPy pour faciliter les calculs
    gamma_bt_array = np.array(gamma_bt_list)
    gamma_bartlett_array = np.array(gamma_bartlett_list)

    # Calculer les statistiques empiriques
    stats_bt = compute_statistics(gamma_bt_array)
    stats_bartlett = compute_statistics(gamma_bartlett_array)

    #calcul de l'eqm
    gamma_th = np.zeros(2*M+1)
    gamma_th[M]=variance
    biais_estim_bt = stats_bt['mean']-gamma_th
    eqm_estim_bt = biais_estim_bt*biais_estim_bt + stats_bt["variance"]
    biais_estim_bartlett = stats_bartlett['mean']-gamma_th
    eqm_estim_bartlett = biais_estim_bartlett*biais_estim_bartlett + stats_bartlett["variance"]


    print("Affichage des résultats pour k = 0")
    print("Blackman-Tukey (k=0):")
    print(f"  Moyenne: {stats_bt['mean'][M]:.3f}")
    print(f"  Variance: {stats_bt['variance'][M]:.3e}")
    
    print("\nBartlett (k=0):")
    print(f"  Moyenne: {stats_bartlett['mean'][M]:.3f}")
    print(f"  Variance: {stats_bartlett['variance'][M]:.3e}")

    print("\nOn constate que l'EQM de Blackman-Tukey avec l'autocorrélation théorique est")
    print("similaire à la variance de Blackman-Tukey. Ici, le biais est négligeable")
    print("devant la variance. Idem pour l'estimateur de Bartlett")

    # Visualisation des biais et de la variance
    k_values = np.arange(-M, M + 1)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(k_values, stats_bt["mean"], label="Moyenne BT", color="blue")
    plt.plot(k_values, stats_bartlett["mean"], label="Moyenne Bartlett", color="orange")
    plt.plot(k_values, gamma_th, label='Gamma théorique', color="red", alpha=0.5)    

    plt.title("Moyenne empirique des coefficients de corrélation")
    plt.xlabel("Indice k")
    plt.ylabel("Moyenne empirique")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(k_values, stats_bt["mean"]-gamma_th, label="Biais BT", color="blue")
    plt.plot(k_values, stats_bartlett["mean"]-gamma_th, label="Biais Bartlett", color="orange")

    plt.title("Biais empirique des coefficients de corrélation")
    plt.xlabel("Indice k")
    plt.ylabel("Biais empirique")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


    
    
    plt.figure(figsize=(10, 10))
    border = 40
    # Premier graphe : Variance
    plt.subplot(2, 1, 1)  # 2 lignes, 1 colonne, 1ère position
    plt.plot(k_values[border:-border], stats_bt["variance"][border:-border], label="Variance BT", color="blue")
    plt.plot(k_values[border:-border], stats_bartlett["variance"][border:-border], label="Variance Bartlett", color="orange")
    plt.title("Variance empirique des coefficients de corrélation sans le bord de "+str(border))
    plt.ylabel("Variance empirique")
    plt.legend()
    plt.grid()

    # Deuxième graphe : EQM
    plt.subplot(2, 1, 2)  # 2 lignes, 1 colonne, 2e position
    plt.plot(k_values[border:-border], eqm_estim_bt[border:-border], label="EQM BT", color="blue")
    plt.plot(k_values[border:-border], eqm_estim_bartlett[border:-border], label="EQM Bartlett", color="orange")
    plt.title("EQM des coefficients de corrélation sans le bord de "+str(border))
    plt.xlabel("Indice k")
    plt.ylabel("EQM avec l'autocorrélation théorique")
    plt.legend()
    plt.grid()

    # Affichage de la figure
    plt.tight_layout()
    plt.show()

def ex4():
    # Paramètres d'entrée
    N = 1024 #Nombre d'échantillons par tranche
    M = 512  # Horizon de calcul
    K = 16    # Nombre de tranches
    variance = 2
    sigma = np.sqrt(variance)
    

    # Génération d'un signal pseudo-aléatoire gaussien
    np.random.seed(42)
    total_signal = sigma * np.random.randn(K * N)  # Signal total (K tranches)

    #definition de la DSP
    def  DSP_estim(gamma, nu):
        dsp = 0
        for k in range(-M,M+1):
            i=k+M
            dsp += gamma[i]*np.exp(-2*1j*np.pi*k*nu)
        return dsp
    

    # Avec k estimations sur les segments

    dsp_bt_list = []
    dsp_bartlett_list = []
    nus = np.linspace(0,1,2*M+1)

    for k in range(K):
        # Extraire la tranche k
        signal = total_signal[k * N:(k + 1) * N]
        
        # Calculer les estimations
        gamma_bt = blackman_tukey_estimator(signal, N, M)
        gamma_bartlett = bartlett_estimator(signal, N, M)
        

        dsp_bt_list.append(DSP_estim(gamma_bt, nus).real)
        dsp_bartlett_list.append(DSP_estim(gamma_bartlett, nus).real)

    dsp_bt = compute_statistics(dsp_bt_list)
    dsp_bartlett = compute_statistics(dsp_bartlett_list)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(nus, dsp_bt['mean'], color="blue", label="DSP avec Blackman-Tukey")
    plt.plot(nus, dsp_bartlett['mean'], color="orange", label="DSP avec Bartlett")
    plt.plot(nus, [variance]*len(nus), color="red", label="DSP théorique")
    plt.title("Estimation de la DSP en divisant en K segments puis moyennant")
    plt.xlabel("Fréquence")
    plt.ylabel("Puissance")
    plt.legend()
    plt.yscale('log')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(nus, dsp_bt['variance'], color="blue", label="DSP avec Blackman-Tukey")
    plt.plot(nus, dsp_bartlett['variance'], color="orange", label="DSP avec Bartlett")
    plt.title("Variance de la DSP en divisant en K segments")
    plt.xlabel("Fréquence")
    plt.ylabel("Puissance")
    plt.legend()
    plt.grid()
    plt.yscale('log')


    plt.tight_layout()
    plt.show()

    print("L'estimation de la DSP par la méthode de Bartlett a une variance plus faible que celle de Blackman Tukey")




if __name__ == "__main__":
    ex1()
    ex2()
    ex3()
    ex4()
