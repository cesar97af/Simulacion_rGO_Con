import matplotlib.pyplot as plt


def plot_transmision(energias, trans_Au, trans_Pd):

    plt.figure(figsize=(10, 5))

    plt.plot(energias, trans_Au, label='Au')

    plt.plot(energias, trans_Pd, label='Pd')

    plt.xlabel("Energía (eV)")

    plt.ylabel("Transmisión")

    plt.legend()

    plt.grid()

    plt.show()


def plot_resistencia(energias, R_Au, R_Pd):

    plt.figure(figsize=(10, 5))

    plt.semilogy(energias, R_Au, label='Au')

    plt.semilogy(energias, R_Pd, label='Pd')

    plt.xlabel("Energía (eV)")

    plt.ylabel("Resistencia (kΩ)")

    plt.legend()

    plt.grid()

    plt.show()