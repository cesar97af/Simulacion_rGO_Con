import kwant
import numpy as np
from scipy import constants


def calcular_transmision(syst, energies):

    transmissions = []

    for E in energies:

        try:

            smatrix = kwant.smatrix(syst, energy=E)

            T = smatrix.transmission(1, 0)

            transmissions.append(T)

        except:

            transmissions.append(0)

    return np.array(transmissions)


def calcular_resistencia_contacto(transmision):

    h = constants.h
    e = constants.e

    R_quantum = h / (2 * e**2)

    T_safe = np.where(transmision > 1e-10, transmision, 1e-10)

    R_contact = R_quantum / T_safe

    return R_contact * 1e-3