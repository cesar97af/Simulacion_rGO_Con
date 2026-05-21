from constantes import D1, D2, epsilon_rGO

def get_rGO_hamiltonian_string():
    """
    Hamiltoniano continuo del rGO
    """

    ham_str = f"""
        {epsilon_rGO} * kron(sigma_0, sigma_0)
        + {D2} * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
        + {D1} * (k_x**4 + k_y**4 + 2*k_x**2*k_y**2) * kron(sigma_0, sigma_0)
        + k_x * kron(sigma_z, sigma_x)
        + k_y * kron(sigma_z, sigma_y)
    """

    return ham_str