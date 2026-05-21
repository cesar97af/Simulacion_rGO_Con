import kwant
import kwant.continuum
import numpy as np

from constantes import *
from hamiltoniano import get_rGO_hamiltonian_string

# 1. Definimos el desorden como una función (Onsite potential dependiente del sitio)
def onsite_disorder(site, disorder_strength=0.2, conversion_factor=1.0):
    """
    Función que Kwant evaluará para cada sitio.
    Multiplica por la matriz identidad del tamaño de los orbitales (norbs=4).
    """
    # Si queremos que el 15% de los sitios tengan desorden:
    # Nota: np.random aquí generará algo dinámico cada vez que se evalúe.
    # Si prefieres desorden congelado (frozen), es mejor definirlo con una semilla fija.
    if np.random.random() < 0.15:
        disorder = disorder_strength * (2 * np.random.random() - 1)
        return disorder * np.eye(4)
    return np.zeros((4, 4))


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

    # --- Construcción del Sistema Central ---
    ham_str = get_rGO_hamiltonian_string()
    template = kwant.continuum.sympify(ham_str)
    discretized = kwant.continuum.discretize(template, grid=a)

    syst = kwant.Builder()

    def shape_rGO(site):
        x, y = site.pos
        return 0 <= x <= L and 0 <= y <= W

    syst.fill(discretized, shape_rGO, (0, 0))

    # AÑADIR DESORDEN: 
    # Recorremos los sitios y sumamos la función de desorden al valor que ya introdujo discretize
    for site in syst.sites():
        # Guardamos el valor previo (que suele ser una función proveniente de continuum.discretize)
        old_value = syst[site]
        
        # Creamos una nueva función que sume el término original y el desorden
        def total_onsite(s, val=old_value):
            # Si el old_value es una función (lo normal en continuum), la evaluamos. 
            # Si es una matriz, se usa directamente.
            base = val(s) if callable(val) else val
            return base + onsite_disorder(s)
            
        syst[site] = total_onsite

    # --- Configuración de los Leads ---
    lat_contact = kwant.lattice.square(a=a, norbs=4)
    lead_onsite = onsite_metal * np.eye(4)
    lead_hopping = -t_interface * np.eye(4)
    
    def shape_lead(pos):
        x, y = pos
        return 0 <= y <= W

    # LEAD IZQUIERDO
    sym_left = kwant.TranslationalSymmetry((-a, 0))
    lead_left = kwant.Builder(sym_left)
    lead_left[lat_contact.shape(shape_lead, (0, W/2))] = lead_onsite
    lead_left[lat_contact.neighbors()] = lead_hopping

    # LEAD DERECHO
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead_right = kwant.Builder(sym_right)
    lead_right[lat_contact.shape(shape_lead, (0, W/2))] = lead_onsite
    lead_right[lat_contact.neighbors()] = lead_hopping

    # --- Conectar Leads al Sistema Central ---
    syst.attach_lead(lead_left)
    syst.attach_lead(lead_right)

    return syst