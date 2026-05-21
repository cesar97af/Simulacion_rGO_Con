import numpy as np

from sistema import crear_sistema_rGO_con_contactos
from analisis import (
    calcular_transmision,
    calcular_resistencia_contacto
)
from visualizacion import (
    plot_transmision,
    plot_resistencia
)

# ============================================================================
# PARÁMETROS
# ============================================================================

L = 20
W = 10
a = 0.5

energias = np.linspace(-2, 2, 41)

# ============================================================================
# SISTEMA Au
# ============================================================================

syst_Au = crear_sistema_rGO_con_contactos(
    L, W, a, metal_type='Au'
).finalized()

trans_Au = calcular_transmision(syst_Au, energias)

R_Au = calcular_resistencia_contacto(trans_Au)

# ============================================================================
# SISTEMA Pd
# ============================================================================

syst_Pd = crear_sistema_rGO_con_contactos(
    L, W, a, metal_type='Pd'
).finalized()

trans_Pd = calcular_transmision(syst_Pd, energias)

R_Pd = calcular_resistencia_contacto(trans_Pd)

# ============================================================================
# GRÁFICAS
# ============================================================================

plot_transmision(
    energias,
    trans_Au,
    trans_Pd
)

plot_resistencia(
    energias,
    R_Au,
    R_Pd
)

print("Simulación completada.")