import numpy as np
import matplotlib.pyplot as plt

def plot_expression(A, N, phi, p0, k_values):
    """
    Trace les valeurs de l'expression donnée en fonction de k.

    Paramètres :
    - A : amplitude
    - N : nombre entier
    - phi : phase en radians
    - p0 : paramètre central
    - k_values : tableau de valeurs de k
    """
    values = []
    for k in k_values:
        term1 = np.exp(1j * phi) * (1 - np.exp(1j * 2 * np.pi * (p0 - k))) / (1 - np.exp(1j * 2 * np.pi * (p0 - k) / N))
        term2 = np.exp(-1j * phi) * (1 - np.exp(-1j * 2 * np.pi * (p0 + k))) / (1 - np.exp(-1j * 2 * np.pi * (p0 + k) / N))
        expression = (A**2 / (4 * N)) * np.abs(term1 - term2)**2
        values.append(expression)

    # Conversion en tableau numpy pour manipulation plus facile
    values = np.array(values)
    
    # Tracé
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, values.real, label="Valeur réelle", color="blue")
    plt.plot(k_values, values.imag, label="Valeur imaginaire", color="orange")
    plt.title("Tracé de l'expression en fonction de k")
    plt.xlabel("k")
    plt.ylabel("Valeur")
    plt.grid(True)
    plt.legend()
    plt.show()

# Exemple d'utilisation
A = 1.0  # Amplitude
N = 100  # Nombre entier
phi = np.pi / 4  # Phase en radians
p0 = 20  # Paramètre central
k_values = range(N) # Valeurs de k
plot_expression(A, N, phi, p0, k_values)