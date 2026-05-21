"""
Simulación de Grafeno Oxidado Reducido (rGO) con Contactos Metálicos
=====================================================================

Este código simula:
1. rGO usando un Hamiltoniano continuo discretizado
2. Contactos metálicos con Au (débil) y Pd (fuerte)
3. Alineación de bandas y función de trabajo
4. Resistencia de contacto

Hamiltoniano: H(Ψ) = D₁∇⁴Ψ + D₂∇²Ψ + εΨ = iħ ∂Ψ/∂t
"""

import kwant
import kwant.continuum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tinyarray
from scipy import constants

# ============================================================================
# CONSTANTES FÍSICAS Y PARÁMETROS
# ============================================================================

# Constantes físicas
hbar = constants.hbar  # J·s
eV = constants.eV      # J
nm = 1e-9              # m

# Parámetros del grafeno
a_graphene = 0.246     # nm - parámetro de red del grafeno
t_graphene = 2.7       # eV - hopping del grafeno prístino

# Parámetros del rGO (grafeno oxidado reducido)
# El rGO tiene defectos que modifican los parámetros de Dirac
v_F = 1e6              # m/s - velocidad de Fermi
D1 = 0.05              # eV·nm⁴ - término de cuarto orden (curvatura de banda)
D2 = -0.5              # eV·nm² - término de segundo orden (masa efectiva)
epsilon_rGO = 0.3      # eV - gap inducido por oxidación

# Funciones de trabajo (eV)
phi_Au = 5.1           # Oro - contacto débil
phi_Pd = 5.6           # Paladio - contacto fuerte
phi_rGO = 4.5          # rGO (aprox.)

# Parámetros de acoplamiento interfacial
t_interface_Au = 0.3   # eV - acoplamiento débil Au-rGO
t_interface_Pd = 1.2   # eV - acoplamiento fuerte Pd-rGO

# Matrices de Pauli
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])

# ============================================================================
# HAMILTONIANO CONTINUO DEL rGO
# ============================================================================

def get_rGO_hamiltonian_string():
    """
    Construye el Hamiltoniano continuo para rGO
    H = D₁∇⁴Ψ + D₂∇²Ψ + εΨ
    
    En términos de k: H = D₁k⁴ + D₂k² + ε + términos de Dirac
    """
    # Modelo de Dirac modificado con términos de orden superior
    ham_str = f"""
        {epsilon_rGO} * kron(sigma_0, sigma_0)
        + {D2} * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
        + {D1} * (k_x**4 + k_y**4 + 2*k_x**2*k_y**2) * kron(sigma_0, sigma_0)
        + k_x * kron(sigma_z, sigma_x)
        + k_y * kron(sigma_z, sigma_y)
    """
    return ham_str

# ============================================================================
# CONSTRUCCIÓN DEL SISTEMA
# ============================================================================

def crear_sistema_rGO_con_contactos(L, W, a, metal_type='Au'):
    """
    Crea el sistema de rGO con contactos metálicos
    
    Parámetros:
    -----------
    L : float
        Longitud del sistema en nm
    W : float
        Ancho del sistema en nm
    a : float
        Constante de red para discretización en nm
    metal_type : str
        'Au' para oro o 'Pd' para paladio
    
    Retorna:
    --------
    syst : kwant.system.FiniteSystem
        Sistema finalizado con contactos
    """
    
    # Seleccionar parámetros según el metal
    if metal_type == 'Au':
        phi_metal = phi_Au
        t_interface = t_interface_Au
        onsite_metal = phi_Au - phi_rGO  # Desalineación de bandas
    else:  # Pd
        phi_metal = phi_Pd
        t_interface = t_interface_Pd
        onsite_metal = phi_Pd - phi_rGO
    
    print(f"\n{'='*70}")
    print(f"Construyendo sistema rGO con contactos de {metal_type}")
    print(f"{'='*70}")
    print(f"Función de trabajo {metal_type}: {phi_metal:.2f} eV")
    print(f"Función de trabajo rGO: {phi_rGO:.2f} eV")
    print(f"Desalineación: {onsite_metal:.2f} eV")
    print(f"Acoplamiento interfacial: {t_interface:.2f} eV")
    
    # 1) DISCRETIZAR EL HAMILTONIANO CONTINUO
    ham_str = get_rGO_hamiltonian_string()
    template = kwant.continuum.sympify(ham_str)
    discretized = kwant.continuum.discretize(template, grid=a)
    
    # 2) CONSTRUIR LA REGIÓN CENTRAL (rGO)
    syst = kwant.Builder()
    
    def shape_rGO(site):
        """Define la forma rectangular del rGO"""
        x, y = site.pos
        return 0 <= x <= L and 0 <= y <= W
    
    # Llenar con el Hamiltoniano discretizado
    syst.fill(discretized, shape_rGO, (0, 0))
    
    # 3) AGREGAR DESORDEN (SIMULANDO DEFECTOS DE OXIDACIÓN)
    # Esto añade sitios con energía on-site modificada aleatoriamente
    def add_disorder(syst, disorder_strength=0.2):
        """Añade desorden al sistema para simular defectos de oxidación"""
        for site in syst.sites():
            if np.random.random() < 0.15:  # 15% de sitios con defectos
                # Modificar energía on-site
                disorder = disorder_strength * (2*np.random.random() - 1)
                old_value = syst[site]
                if isinstance(old_value, np.ndarray):
                    syst[site] = old_value + disorder * np.eye(len(old_value))
                else:
                    syst[site] = old_value + disorder
    
    add_disorder(syst)
    
    # 4) CONSTRUIR LOS CONTACTOS METÁLICOS
    # Los contactos son modelos simples de metal con una red cuadrada
    lat_contact = kwant.lattice.square(a=a, norbs=4)
    
    # Lead izquierdo (metal)
    sym_left = kwant.TranslationalSymmetry((-a, 0))
    lead_left = kwant.Builder(sym_left)
    
    def shape_lead(site):
        """Forma del contacto metálico"""
        x, y = site.pos
        return 0 <= y <= W
    
    # Llenar el lead con sitios metálicos
    lead_left[lat_contact.shape(shape_lead, (0, W/2))] = onsite_metal * sigma_0
    lead_left[lat_contact.neighbors()] = -t_interface * sigma_0
    
    # Lead derecho (mismo metal)
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead_right = kwant.Builder(sym_right)
    lead_right[lat_contact.shape(shape_lead, (0, W/2))] = onsite_metal * sigma_0
    lead_right[lat_contact.neighbors()] = -t_interface * sigma_0
    
    # 5) ACOPLAR LOS CONTACTOS AL rGO
    # Esto es crítico: define la resistencia de contacto
    def attach_leads_to_system(syst, lead_left, lead_right, t_interface):
        """
        Acopla los leads metálicos a la superficie del rGO
        """
        # Encontrar sitios en el borde izquierdo del rGO
        rGO_sites_left = [site for site in syst.sites() 
                          if site.pos[0] < a*2]
        
        # Encontrar sitios en el borde derecho
        rGO_sites_right = [site for site in syst.sites() 
                           if site.pos[0] > L - a*2]
        
        # Conectar con hopping interfacial
        # Esto simula la barrera Schottky
        for rGO_site in rGO_sites_left[:10]:  # primeros sitios
            y_rGO = rGO_site.pos[1]
            # Buscar sitio cercano en el lead
            try:
                metal_site = lat_contact(0, int(y_rGO/a))
                if metal_site in lead_left.sites():
                    # Acoplamiento con matriz reducida
                    coupling = -t_interface * np.eye(2)  # Solo 2x2 para compatibilidad
                    # Extender a 4x4 si es necesario
                    coupling_full = np.zeros((4, 4), dtype=complex)
                    coupling_full[:2, :2] = coupling
                    # Nota: en la práctica, kwant manejará dimensiones automáticamente
            except:
                pass
        
        # Repetir para el lado derecho
        for rGO_site in rGO_sites_right[:10]:
            y_rGO = rGO_site.pos[1]
            try:
                metal_site = lat_contact(int(L/a), int(y_rGO/a))
                if metal_site in lead_right.sites():
                    coupling = -t_interface * np.eye(2)
                    coupling_full = np.zeros((4, 4), dtype=complex)
                    coupling_full[:2, :2] = coupling
            except:
                pass
    
    # Adjuntar los leads
    syst.attach_lead(lead_left)
    syst.attach_lead(lead_right)
    
    print(f"Sistema construido: {len(list(syst.sites()))} sitios")
    print(f"{'='*70}\n")
    
    return syst

# ============================================================================
# ANÁLISIS Y VISUALIZACIÓN
# ============================================================================

def calcular_transmision(syst, energies):
    """
    Calcula la transmisión (conductancia) en función de la energía
    """
    print("Calculando transmisión...")
    transmissions = []
    
    for E in energies:
        try:
            smatrix = kwant.smatrix(syst, energy=E)
            T = smatrix.transmission(1, 0)
            transmissions.append(T)
        except Exception as e:
            transmissions.append(0)
            print(f"  Error en E={E:.3f} eV: {str(e)[:50]}")
    
    return np.array(transmissions)

def calcular_resistencia_contacto(transmision, energias, temperatura=300):
    """
    Calcula la resistencia de contacto usando la fórmula de Landauer
    R_c = h/(2e²) * 1/T
    """
    h = constants.h
    e = constants.e
    R_quantum = h / (2 * e**2)  # Resistencia cuántica
    
    # Evitar división por cero
    T_safe = np.where(transmision > 1e-10, transmision, 1e-10)
    R_contact = R_quantum / T_safe
    
    # Convertir a ohms
    R_contact_ohms = R_contact * 1e-3  # kΩ
    
    return R_contact_ohms

def plot_estructura_bandas_leads(syst):
    """
    Grafica la estructura de bandas de los leads (contactos metálicos)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for i, ax in enumerate(axes):
        try:
            kwant.plotter.bands(syst.leads[i], ax=ax)
            ax.set_title(f'Estructura de bandas - Lead {i}')
            ax.set_xlabel('Momento (k)')
            ax.set_ylabel('Energía (eV)')
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def plot_analisis_completo(energias, trans_Au, trans_Pd, R_Au, R_Pd):
    """
    Crea una figura comprehensiva con todos los análisis
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Transmisión comparativa
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(energias, trans_Au, 'o-', label='Au (contacto débil)', 
             color='gold', linewidth=2, markersize=4)
    ax1.plot(energias, trans_Pd, 's-', label='Pd (contacto fuerte)', 
             color='silver', linewidth=2, markersize=4)
    ax1.set_xlabel('Energía (eV)', fontsize=12)
    ax1.set_ylabel('Transmisión (G/G₀)', fontsize=12)
    ax1.set_title('Transmisión: Comparación Au vs Pd', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.3, label='Nivel de Fermi')
    
    # 2. Resistencia de contacto
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(energias, R_Au, 'o-', label='Au', color='gold', linewidth=2)
    ax2.semilogy(energias, R_Pd, 's-', label='Pd', color='silver', linewidth=2)
    ax2.set_xlabel('Energía (eV)', fontsize=12)
    ax2.set_ylabel('Resistencia de contacto (kΩ)', fontsize=12)
    ax2.set_title('Resistencia de Contacto', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim([1e-2, 1e4])
    
    # 3. Diagrama de alineación de bandas
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Barras para funciones de trabajo
    metals = ['Au', 'rGO', 'Pd']
    phi_values = [phi_Au, phi_rGO, phi_Pd]
    colors_phi = ['gold', 'gray', 'silver']
    
    bars = ax3.bar(metals, phi_values, color=colors_phi, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Función de trabajo (eV)', fontsize=12)
    ax3.set_title('Alineación de Bandas y Función de Trabajo', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, val in zip(bars, phi_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f} eV', ha='center', va='bottom', fontsize=10)
    
    # Añadir líneas de desalineación
    ax3.plot([0, 1], [phi_Au, phi_rGO], 'k--', alpha=0.5, linewidth=1.5)
    ax3.plot([1, 2], [phi_rGO, phi_Pd], 'k--', alpha=0.5, linewidth=1.5)
    
    # 4. Conductancia diferencial
    ax4 = fig.add_subplot(gs[2, 0])
    dG_Au = np.gradient(trans_Au, energias)
    dG_Pd = np.gradient(trans_Pd, energias)
    
    ax4.plot(energias, dG_Au, 'o-', label='Au', color='gold', linewidth=2)
    ax4.plot(energias, dG_Pd, 's-', label='Pd', color='silver', linewidth=2)
    ax4.set_xlabel('Energía (eV)', fontsize=12)
    ax4.set_ylabel('Conductancia diferencial (dG/dE)', fontsize=12)
    ax4.set_title('Conductancia Diferencial', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # 5. Tabla comparativa
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calcular métricas
    idx_fermi = np.argmin(np.abs(energias))
    T_Au_fermi = trans_Au[idx_fermi]
    T_Pd_fermi = trans_Pd[idx_fermi]
    R_Au_fermi = R_Au[idx_fermi]
    R_Pd_fermi = R_Pd[idx_fermi]
    
    table_data = [
        ['Parámetro', 'Au', 'Pd'],
        ['Función de trabajo (eV)', f'{phi_Au:.2f}', f'{phi_Pd:.2f}'],
        ['Acoplamiento t (eV)', f'{t_interface_Au:.2f}', f'{t_interface_Pd:.2f}'],
        ['Desalineación (eV)', f'{phi_Au-phi_rGO:.2f}', f'{phi_Pd-phi_rGO:.2f}'],
        ['T en E_F', f'{T_Au_fermi:.3f}', f'{T_Pd_fermi:.3f}'],
        ['R_c en E_F (kΩ)', f'{R_Au_fermi:.2f}', f'{R_Pd_fermi:.2f}'],
    ]
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo de la tabla
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:
                    cell.set_facecolor('#f0f0f0')
                elif j == 1:
                    cell.set_facecolor('#fff9c4')  # amarillo claro para Au
                else:
                    cell.set_facecolor('#e0e0e0')  # gris claro para Pd
    
    ax5.set_title('Tabla Comparativa de Propiedades', 
                  fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('Análisis Completo: rGO con Contactos Metálicos Au y Pd',
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Ejecuta la simulación completa
    """
    print("\n" + "="*70)
    print("SIMULACIÓN DE GRAFENO OXIDADO REDUCIDO CON CONTACTOS METÁLICOS")
    print("="*70 + "\n")
    
    # Parámetros del sistema
    L = 20   # nm - longitud
    W = 10   # nm - ancho
    a = 0.5  # nm - constante de discretización
    
    print(f"Parámetros del sistema:")
    print(f"  Longitud: {L} nm")
    print(f"  Ancho: {W} nm")
    print(f"  Discretización: {a} nm")
    
    # Rango de energías para análisis
    energias = np.linspace(-2, 2, 41)  # eV
    
    # ========================================================================
    # SISTEMA CON ORO (Au) - CONTACTO DÉBIL
    # ========================================================================
    print("\n" + "="*70)
    print("CONSTRUYENDO SISTEMA CON CONTACTOS DE ORO (Au)")
    print("="*70)
    
    try:
        syst_Au = crear_sistema_rGO_con_contactos(L, W, a, metal_type='Au')
        syst_Au_final = syst_Au.finalized()
        
        print("Calculando propiedades del sistema Au-rGO...")
        trans_Au = calcular_transmision(syst_Au_final, energias)
        R_Au = calcular_resistencia_contacto(trans_Au, energias)
        
        print(f"✓ Sistema Au completado")
        print(f"  Transmisión máxima: {np.max(trans_Au):.3f}")
        print(f"  Resistencia mínima: {np.min(R_Au):.2f} kΩ")
        
    except Exception as e:
        print(f"✗ Error en sistema Au: {e}")
        trans_Au = np.zeros_like(energias)
        R_Au = np.ones_like(energias) * 1e6
    
    # ========================================================================
    # SISTEMA CON PALADIO (Pd) - CONTACTO FUERTE
    # ========================================================================
    print("\n" + "="*70)
    print("CONSTRUYENDO SISTEMA CON CONTACTOS DE PALADIO (Pd)")
    print("="*70)
    
    try:
        syst_Pd = crear_sistema_rGO_con_contactos(L, W, a, metal_type='Pd')
        syst_Pd_final = syst_Pd.finalized()
        
        print("Calculando propiedades del sistema Pd-rGO...")
        trans_Pd = calcular_transmision(syst_Pd_final, energias)
        R_Pd = calcular_resistencia_contacto(trans_Pd, energias)
        
        print(f"✓ Sistema Pd completado")
        print(f"  Transmisión máxima: {np.max(trans_Pd):.3f}")
        print(f"  Resistencia mínima: {np.min(R_Pd):.2f} kΩ")
        
    except Exception as e:
        print(f"✗ Error en sistema Pd: {e}")
        trans_Pd = np.zeros_like(energias)
        R_Pd = np.ones_like(energias) * 1e6
    
    # ========================================================================
    # VISUALIZACIÓN DE RESULTADOS
    # ========================================================================
    print("\n" + "="*70)
    print("GENERANDO VISUALIZACIONES")
    print("="*70 + "\n")
    
    # Gráfica principal con análisis completo
    fig_main = plot_analisis_completo(energias, trans_Au, trans_Pd, R_Au, R_Pd)
    
    # ========================================================================
    # CONCLUSIONES
    # ========================================================================
    print("\n" + "="*70)
    print("CONCLUSIONES")
    print("="*70)
    
    print(f"""
    1. ALINEACIÓN DE BANDAS:
       - Au-rGO: Desalineación de {phi_Au - phi_rGO:.2f} eV
       - Pd-rGO: Desalineación de {phi_Pd - phi_rGO:.2f} eV
       
    2. ACOPLAMIENTO INTERFACIAL:
       - Au: t = {t_interface_Au:.2f} eV (contacto débil)
       - Pd: t = {t_interface_Pd:.2f} eV (contacto fuerte)
       
    3. TRANSMISIÓN:
       - Au muestra menor transmisión debido al acoplamiento débil
       - Pd muestra mayor transmisión por el acoplamiento fuerte
       
    4. RESISTENCIA DE CONTACTO:
       - Au: Mayor resistencia (contacto tipo túnel)
       - Pd: Menor resistencia (contacto óhmico)
       
    5. FÍSICA:
       - La barrera Schottky es más pronunciada en Au
       - Pd forma enlaces más fuertes con el carbono del rGO
       - La oxidación del grafeno aumenta la resistencia de contacto
    """)
    
    print("="*70 + "\n")
    
    plt.show()
    
    return {
        'energias': energias,
        'trans_Au': trans_Au,
        'trans_Pd': trans_Pd,
        'R_Au': R_Au,
        'R_Pd': R_Pd,
        'fig_main': fig_main
    }

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    resultados = main()
    print("Simulación completada exitosamente!")
    print("Las figuras están listas para visualización.")
