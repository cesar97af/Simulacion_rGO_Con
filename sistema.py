import kwant
import kwant.continuum
import numpy as np

from constantes import *
from hamiltoniano import get_rGO_hamiltonian_string


def add_disorder(syst, disorder_strength=0.2):

    for site in syst.sites():

        if np.random.random() < 0.15:

            disorder = disorder_strength * (2*np.random.random() - 1)

            old_value = syst[site]

            if isinstance(old_value, np.ndarray):
                syst[site] = old_value + disorder * np.eye(len(old_value))
            else:
                syst[site] = old_value + disorder


def crear_sistema_rGO_con_contactos(L, W, a, metal_type='Au'):

    if metal_type == 'Au':

        phi_metal = phi_Au
        t_interface = t_interface_Au
        onsite_metal = phi_Au - phi_rGO

    else:

        phi_metal = phi_Pd
        t_interface = t_interface_Pd
        onsite_metal = phi_Pd - phi_rGO

    print(f"\nConstruyendo sistema con {metal_type}")

    ham_str = get_rGO_hamiltonian_string()

    template = kwant.continuum.sympify(ham_str)

    discretized = kwant.continuum.discretize(template, grid=a)

    syst = kwant.Builder()

    def shape_rGO(site):

        x, y = site.pos

        return 0 <= x <= L and 0 <= y <= W

    syst.fill(discretized, shape_rGO, (0, 0))

    add_disorder(syst)

    lat_contact = kwant.lattice.square(a=a, norbs=4)

    # =========================================================================
    # LEAD IZQUIERDO
    # =========================================================================

    sym_left = kwant.TranslationalSymmetry((-a, 0))

    lead_left = kwant.Builder(sym_left)

    def shape_lead(site):

        x, y = site.pos

        return 0 <= y <= W

    lead_left[lat_contact.shape(shape_lead, (0, W/2))] = onsite_metal * sigma_0

    lead_left[lat_contact.neighbors()] = -t_interface * sigma_0

    # =========================================================================
    # LEAD DERECHO
    # =========================================================================

    sym_right = kwant.TranslationalSymmetry((a, 0))

    lead_right = kwant.Builder(sym_right)

    lead_right[lat_contact.shape(shape_lead, (0, W/2))] = onsite_metal * sigma_0

    lead_right[lat_contact.neighbors()] = -t_interface * sigma_0

    syst.attach_lead(lead_left)

    syst.attach_lead(lead_right)

    return syst