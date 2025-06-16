import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from hashlib import md5
from copy import deepcopy
import inspect

# Global cache for mode solutions
_mode_cache = {}

def circular_circumference(radius):
    """Calculate the circumference of a circular ring.

    Parameters:
        radius (float): Radius of the ring (m).

    Returns:
        float: Circumference (m).
    """
    return 2 * np.pi * radius

def elliptical_circumference(a, b):
    """Approximate the circumference of an elliptical ring using Ramanujan's formula.

    Parameters:
        a (float): Semi-major axis (m).
        b (float): Semi-minor axis (m).

    Returns:
        float: Circumference (m).
    """
    h = ((a - b)**2) / ((a + b)**2)
    return np.pi * (a + b) * (1 + (3*h) / (10 + np.sqrt(4 - 3*h)))

def racetrack_circumference(radius, straight_length):
    """Calculate the circumference of a racetrack resonator.

    Parameters:
        radius (float): Radius of curved sections (m).
        straight_length (float): Length of straight segments (m).

    Returns:
        float: Circumference (m).
    """
    return 2 * np.pi * radius + 2 * straight_length

def get_circumference(resonator_type, **geometry_params):
    """Get the circumference based on resonator type.

    Parameters:
        resonator_type (str): 'circular', 'elliptical', or 'racetrack'.
        geometry_params: Parameters like radius, a, b, straight_length (m).

    Returns:
        float: Circumference (m).

    Raises:
        ValueError: If resonator_type or parameters are invalid.
    """
    if resonator_type == 'circular':
        if 'radius' not in geometry_params:
            raise ValueError("Circular resonator requires 'radius' parameter")
        return circular_circumference(geometry_params['radius'])
    elif resonator_type == 'elliptical':
        if 'a' not in geometry_params or 'b' not in geometry_params:
            raise ValueError("Elliptical resonator requires 'a' and 'b' parameters")
        return elliptical_circumference(geometry_params['a'], geometry_params['b'])
    elif resonator_type == 'racetrack':
        if 'radius' not in geometry_params or 'straight_length' not in geometry_params:
            raise ValueError("Racetrack resonator requires 'radius' and 'straight_length' parameters")
        return racetrack_circumference(geometry_params['radius'], geometry_params['straight_length'])
    else:
        raise ValueError(f"Unsupported resonator_type: {resonator_type}")

def solve_waveguide_modes(wl, layers, T, polarization='TE', num_modes=1, dx=10e-9, disable_cache=False):
    """Solve the vector wave equation for waveguide modes using FDM with PML.

    Parameters:
        wl (float or array): Wavelength (m).
        layers (list): List of dicts with material (str), thickness (m), n (float), dn_dT (1/°C), alpha (1/m or callable).
        T (float): Temperature (°C).
        polarization (str): 'TE' or 'TM'.
        num_modes (int): Number of modes to compute.
        dx (float): Grid spacing (m, default=10 nm).
        disable_cache (bool): Disable caching for mode solutions (default=False).

    Returns:
        tuple: (E_field, n_eff)
            - E_field: Mode profiles (shape: [len(wl), num_modes, N_grid] or [num_modes, N_grid]).
            - n_eff: Effective array (shape: [len(wl), num_modes] or [num_modes]).

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If eigenvalue solver fails.
    """
    if not layers:
        raise ValueError("Layers list cannot be empty")
    if dx <= 0:
        raise ValueError("Grid spacing dx must be positive")
    if num_modes < 1:
        raise ValueError("Number of modes must be at least 1")
    if polarization not in ['TE', 'TM']:
        raise ValueError("Invalid polarization")
    
    wl = np.asarray(wl)
    scalar_wl = wl.ndim == 0
    if scalar_wl:
        wl = np.array([wl])
    if np.any(wl <= 0):
        raise ValueError("Wavelength must be positive")

    # Precompute layer boundaries
    boundaries = np.cumsum([0] + [layer['thickness'] for layer in layers])
    total_thickness = boundaries[-1]
    N = int(total_thickness / dx) + 1
    x = np.linspace(0, total_thickness, N)
    dx = x[1] - x[0]

    # PML parameters
    pml_thickness = 10 * dx
    sigma_max = 1e6  # PML conductivity
    x_pml = np.concatenate([np.linspace(-pml_thickness, 0, 50)[::-1], x, np.linspace(total_thickness, total_thickness + pml_thickness, 50)])
    N_total = len(x_pml)
    s_x = np.ones(N_total, dtype=np.complex128)
    for i in range(50):
        d = i / 49 * pml_thickness
        s_x[i] = 1 - 1j * sigma_max * (d / pml_thickness)**2 / (2 * np.pi / wl[0])
        s_x[-(i+1)] = s_x[i]

    # Cache key
    def get_layer_key(layer):
        if callable(layer['alpha']):
            try:
                return inspect.getsource(layer['alpha'])
            except OSError:
                return id(layer['alpha'])  # Fallback for uninspectable callables
        return layer['alpha']
    layers_str = str([(l['material'], l['thickness'], l['n'], l['dn_dT'], get_layer_key(l)) for l in layers])
    cache_key_base = md5(f"{layers_str}_{T}_{polarization}_{dx}_{num_modes}".encode()).hexdigest()

    E_field_all = np.zeros((len(wl), num_modes, N_total), dtype=np.complex128)
    n_eff_all = np.zeros((len(wl), num_modes))

    for i, lambda_i in enumerate(wl):
        cache_key = f"{cache_key_base}_{lambda_i}"
        if not disable_cache and cache_key in _mode_cache:
            E_field_all[i], n_eff_all[i] = _mode_cache[cache_key]
            continue

        k0 = 2 * np.pi / lambda_i
        n = np.zeros(N_total)
        n_pml = np.ones(N_total)
        for j in range(len(layers)):
            mask = (x_pml >= boundaries[j]) & (x_pml < boundaries[j+1])
            n_layer = layers[j]['n'] + layers[j]['dn_dT'] * (T - 25)
            n[mask] = n_layer
            n_pml[mask] = n_layer**2

        try:
            # Build FDM matrix with stretched coordinates
            s_x_center = s_x[1:-1]
            s_x_left = (s_x[:-2] + s_x_center) / 2
            s_x_right = (s_x[2:] + s_x_center) / 2

            if polarization == 'TE':
                diag = (k0 * n[1:-1])**2 - (1/(dx**2 * s_x_center)) * (1/s_x_left + 1/s_x_right)
                off_diag = 1 / (dx**2 * np.sqrt(s_x_left * s_x_right))
                A = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(N_total-2, N_total-2), format='csr')
                beta2, E_field = spla.eigsh(A, k=num_modes, which='LM')
                field = E_field
            else:  # TM
                inv_n2 = 1 / n_pml[1:-1]
                A_diag = k0**2 * inv_n2 - (1/(dx**2 * s_x_center)) * (inv_n2/s_x_left + inv_n2/s_x_right)
                A_off_diag = inv_n2 / (dx**2 * np.sqrt(s_x_left * s_x_right))
                B_diag = inv_n2
                A = sp.diags([A_off_diag, A_diag, A_off_diag], [-1, 0, 1], shape=(N_total-2, N_total-2), format='csr')
                B = sp.diags(B_diag, format='csr')
                beta2, H_field = spla.eigsh(A, k=num_modes, M=B, which='LM', sigma=0)
                field = H_field

            beta = np.sqrt(np.abs(beta2))
            n_eff = beta / k0

            # Pad field with zeros at boundaries
            field_padded = np.zeros((N_total, num_modes), dtype=np.complex128)
            field_padded[1:-1, :] = field

            # Normalize modes
            for j in range(num_modes):
                field_padded[:, j] /= np.sqrt(np.trapz(np.abs(field_padded[:, j])**2, x=x_pml))

            # Sort by descending n_eff
            idx = np.argsort(n_eff)[::-1]
            n_eff = n_eff[idx]
            field_padded = field_padded[:, idx]

            E_field_all[i] = field_padded.T
            n_eff_all[i] = n_eff

            # Cache result
            if not disable_cache:
                _mode_cache[cache_key] = (field_padded.T, n_eff)

        except Exception as e:
            raise RuntimeError(f"Eigenvalue solver failed for wavelength {lambda_i:.2e} m: {str(e)}")

    if scalar_wl:
        return E_field_all[0], n_eff_all[0]
    return E_field_all, n_eff_all

def compute_group_index(wl, n_eff):
    """Calculate the group index from effective refractive index dispersion.

    Parameters:
        wl (array): Wavelength array (m).
        n_eff (array): Effective refractive index array (same shape as wl).

    Returns:
        array: Group index array.

    Raises:
        ValueError: If shapes mismatch or wl is not monotonically increasing.

    Equation:
        n_g = n_eff - λ * (d n_eff / d λ)
    """
    wl = np.asarray(wl)
    n_eff = np.asarray(n_eff)
    if wl.shape != n_eff.shape:
        raise ValueError("Wavelength and n_eff arrays must have the same shape")
    if len(wl) < 2:
        raise ValueError("Wavelength array must have at least two points")

    # Ensure wl is unique and monotonically increasing
    wl_unique, idx = np.unique(wl, return_index=True)
    if len(wl_unique) < len(wl):
        n_eff = n_eff[idx]
        wl = wl_unique
    if not np.all(np.diff(wl) > 0):
        raise ValueError("Wavelength array must be strictly monotonically increasing")

    # Compute differences and ensure minimum spacing
    min_dw = 1e-12  # Minimum wavelength difference (m)
    d_lambda = np.gradient(wl)
    d_lambda = np.maximum(d_lambda, min_dw)  # Prevent division by zero
    d_n_eff = np.gradient(n_eff, d_lambda)
    n_g = n_eff - wl * d_n_eff
    return n_g

def compute_layered_alpha_from_mode(layers, wl, E_field, dx):
    """Compute effective loss coefficient based on mode confinement.

    Parameters:
        layers (list): List of dicts with material, thickness (m), alpha (1/m or callable).
        wl (array): Wavelength array (m).
        E_field (array): Mode profile (shape: [N_grid]).
        dx (float): Grid spacing (m).

    Returns:
        array: Effective loss coefficient (1/m).

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        α_eff = Σ α_i * (∫_layer_i |E|^2 dx) / (∫ |E|^2 dx)
    """
    if not layers:
        raise ValueError("Layers list cannot be empty")
    boundaries = np.cumsum([0] + [layer['thickness'] for layer in layers])
    alpha_eff = np.zeros_like(wl)
    power = np.trapz(np.abs(E_field)**2, dx=dx)
    if power == 0:
        raise ValueError("Mode power is zero")
    x = np.arange(len(E_field)) * dx
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or not all(k in layer for k in ['material', 'thickness', 'alpha']):
            raise ValueError("Each layer must have 'material', 'thickness', and 'alpha'")
        mask = (x >= boundaries[i]) & (x < boundaries[i+1])
        alpha = evaluate_alpha(layer['alpha'], wl)
        layer_power = np.trapz(np.abs(E_field[mask])**2, dx=dx)
        alpha_eff += alpha * layer_power / power
    return alpha_eff

def propagation_phase(n_eff, wavelength, length):
    """Compute the complex phase factor exp(-j β L).

    Parameters:
        n_eff (float or array): Effective refractive index.
        wavelength (float or array): Wavelength (m).
        length (float): Propagation length (m).

    Returns:
        array: Complex phase factor.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        exp(-j β L), β = 2π n_eff / λ
    """
    if np.any(wavelength <= 0):
        raise ValueError("Wavelength must be positive")
    if length < 0:
        raise ValueError("Length must be non-negative")
    beta = 2 * np.pi * n_eff / wavelength
    return np.exp(-1j * beta * length)

def evaluate_alpha(alpha, wl):
    """Evaluate alpha, which can be a scalar, array, or callable.

    Parameters:
        alpha (float, array, or callable): Loss coefficient (1/m).
        wl (array): Wavelength array (m).

    Returns:
        array: Loss coefficient array (1/m).

    Raises:
        ValueError: If alpha is invalid.
    """
    if callable(alpha):
        alpha_val = alpha(wl)
    elif isinstance(alpha, (int, float)):
        alpha_val = np.full_like(wl, float(alpha))
    else:
        alpha_val = np.asarray(alpha)
        if alpha_val.shape != wl.shape:
            raise ValueError("Alpha array must match wavelength array shape")
    if np.any(alpha_val < 0):
        raise ValueError("Loss coefficient alpha must be non-negative")
    return alpha_val

# -------------------------
# Single Components
# -------------------------
def single_waveguide(wl, layers, T, polarization, length):
    """Compute complex field and power transmission for a straight waveguide.

    Parameters:
        wl (array): Wavelength array (m).
        layers (list): List of dicts with material, thickness (m), n, dn_dT (1/°C), alpha (1/m or callable).
        T (float): Temperature (°C).
        polarization (str): 'TE' or 'TM'.
        length (float): Waveguide length (m).

    Returns:
        tuple: (E, T_power)
            - E: Complex field transmission.
            - T_power: Power transmission.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        E = exp(-α L / 2) exp(-j β L)
    """
    if length < 0:
        raise ValueError("Waveguide length must be non-negative")
    E_field, n_eff = solve_waveguide_modes(wl, layers, T, polarization, num_modes=1)
    n_eff = n_eff[:, 0]  # Fundamental mode
    total_thickness = sum(layer['thickness'] for layer in layers)
    dx = total_thickness / (E_field.shape[2] - 1)
    alpha = compute_layered_alpha_from_mode(layers, wl, E_field[0, 0], dx)
    phi = propagation_phase(n_eff, wl, length)
    A = np.exp(-alpha * length / 2)
    E = A * phi
    return E, np.abs(E)**2

def single_mrr_thru(wl, resonator_type, layers, T, polarization, t, **geometry_params):
    """Compute through-port transmission for a single all-pass MRR.

    Parameters:
        wl (array): Wavelength array (m).
        resonator_type (str): 'circular', 'elliptical', or 'racetrack'.
        layers (list): List of dicts with material, thickness (m), n, dn_dT (1/°C), alpha (1/m or callable).
        T (float): Temperature (°C).
        polarization (str): 'TE' or 'TM'.
        t (complex): Through-coupling coefficient.
        geometry_params: Parameters like radius, a, b, straight_length (m).

    Returns:
        tuple: (H, T_power)
            - H: Complex transfer function.
            - T_power: Power transmission.

    Raises:
        ValueError: If inputs are invalid.

    Equation:
        H = (t - A exp(-j φ)) / (1 - t A exp(-j φ))
    """
    if not 0 <= np.abs(t) <= 1:
        raise ValueError("Coupling coefficient t must satisfy 0 <= |t| <= 1")
    L = get_circumference(resonator_type, **geometry_params)
    E_field, n_eff = solve_waveguide_modes(wl, layers, T, polarization, num_modes=1)
    n_eff = n_eff[:, 0]
    total_thickness = sum(layer['thickness'] for layer in layers)
    dx = total_thickness / (E_field.shape[2] - 1)
    alpha = compute_layered_alpha_from_mode(layers, wl, E_field[0, 0], dx)
    A_round = np.exp(-alpha * L)
    phi = 2 * np.pi * n_eff * L / wl
    E_rt = A_round * np.exp(-1j * phi)
    H = (t - E_rt) / (1 - t * E_rt)
    return H, np.abs(H)**2

def single_mrr_add_drop(wl, resonator_type, layers, T, polarization, t1, kappa1, t2=None, kappa2=None, **geometry_params):
    """Compute through and drop port transmissions for an add-drop MRR.

    Parameters:
        wl (array): Wavelength array (m).
        resonator_type (str): 'circular', 'elliptical', or 'racetrack'.
        layers (list): List of dicts with material, thickness (m), n, dn_dT (1/°C), alpha (1/m or callable).
        T (float): Temperature (°C).
        polarization (str): 'TE' or 'TM'.
        t1, kappa1 (complex): Input coupler coefficients (|t1|^2 + |kappa1|^2 = 1).
        t2, kappa2 (complex): Output coupler coefficients (default to t1, kappa1).
        geometry_params: Parameters like radius, a, b, straight_length (m).

    Returns:
        tuple: ((H_thru, H_drop), (T_thru, T_drop))
            - H_thru, H_drop: Complex transfer functions.
            - T_thru, T_drop: Power transmissions.

    Raises:
        ValueError: If inputs are invalid.

    Equations:
        H_thru = (t1 - t2 A exp(-j φ)) / (1 - t1 t2 A exp(-j φ))
        H_drop = -κ1 κ2 sqrt(A) exp(-j φ/2) / (1 - t1 t2 A exp(-j φ))
    """
    if not np.isclose(np.abs(t1)**2 + np.abs(kappa1)**2, 1, atol=1e-6):
        raise ValueError("Input coupler must satisfy |t1|^2 + |kappa1|^2 = 1")
    t2 = t1 if t2 is None else t2
    kappa2 = kappa1 if kappa2 is None else kappa2
    if not np.isclose(np.abs(t2)**2 + np.abs(kappa2)**2, 1, atol=1e-6):
        raise ValueError("Output coupler must satisfy |t2|^2 + |kappa2|^2 = 1")
    L = get_circumference(resonator_type, **geometry_params)
    E_field, n_eff = solve_waveguide_modes(wl, layers, T, polarization, num_modes=1)
    n_eff = n_eff[:, 0]
    total_thickness = sum(layer['thickness'] for layer in layers)
    dx = total_thickness / (E_field.shape[2] - 1)
    alpha = compute_layered_alpha_from_mode(layers, wl, E_field[0, 0], dx)
    A_round = np.exp(-alpha * L)
    A_half = np.exp(-alpha * L / 2)
    phi = 2 * np.pi * n_eff * L / wl
    E_rt = A_round * np.exp(-1j * phi)
    H_thru = (t1 - t2 * E_rt) / (1 - t1 * t2 * E_rt)
    H_drop = -kappa1 * kappa2 * np.sqrt(A_half) * np.exp(-1j * phi / 2) / (1 - t1 * t2 * E_rt)
    return (H_thru, H_drop), (np.abs(H_thru)**2, np.abs(H_drop)**2)

# -------------------------
# Cascaded MRRs
# -------------------------
def cascaded_mrrs_add_drop(wl, params_list, bus_segments=None):
    """Compute transmission for cascaded add-drop MRRs.

    Parameters:
        wl (array): Wavelength array (m).
        params_list (list): List of dicts with resonator_type, layers, T (°C), polarization,
                            t1, kappa1, t2, kappa2 (complex), geometry parameters.
        bus_segments (list, optional): List of dicts with length (m), alpha (1/m or callable), n_eff.

    Returns:
        tuple: (E_thru, T_thru, drop_powers)
            - E_thru: Final through-port complex field.
            - T_thru: Final through-port power.
            - drop_powers: List of drop-port power arrays.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not params_list:
        raise ValueError("params_list cannot be empty")
    if bus_segments and len(bus_segments) != len(params_list) - 1:
        raise ValueError("Number of bus segments must be N-1 for N rings")
    bus_field = np.ones_like(wl, dtype=complex)
    drop_powers = []
    for i, p in enumerate(params_list):
        (H_thru, H_drop), (T_thru, T_drop) = single_mrr_add_drop(
            wl, p['resonator_type'], p['layers'], p['T'], p['polarization'],
            p['t1'], p['kappa1'], p.get('t2', p['t1']), p.get('kappa2', p['kappa1']),
            **{k: p[k] for k in p if k in ['radius', 'a', 'b', 'straight_length']}
        )
        drop_powers.append(np.abs(bus_field * H_drop)**2)
        bus_field = bus_field * H_thru
        if bus_segments and i < len(params_list) - 1:
            seg = bus_segments[i]
            if seg['length'] < 0:
                raise ValueError("Bus segment length must be non-negative")
            alpha_bus = evaluate_alpha(seg['alpha'], wl)
            bus_phi = 2 * np.pi * seg['n_eff'] * seg['length'] / wl
            bus_A = np.exp(-alpha_bus * seg['length'] / 2)
            bus_field *= bus_A * np.exp(-1j * bus_phi)
    return bus_field, np.abs(bus_field)**2, drop_powers

# -------------------------
# Selector Function
# -------------------------
def compute_transmission(case, **kwargs):
    """Select and compute transmission based on the case.

    Parameters:
        case (str): 'waveguide', 'mrr_thru', 'mrr_add_drop', or 'multi_mrr'.
        kwargs: Case-specific parameters (wavelengths, layers, T, polarization, etc.).

    Returns:
        Output from the corresponding function.

    Raises:
        ValueError: If inputs are invalid.
    """
    wl = np.asarray(kwargs['wavelengths'])
    if np.any(wl <= 0):
        raise ValueError("Wavelengths must be positive")
    if case == 'waveguide':
        return single_waveguide(wl, kwargs['layers'], kwargs['T'], kwargs['polarization'], kwargs['length'])
    elif case == 'mrr_thru':
        return single_mrr_thru(
            wl, kwargs['resonator_type'], kwargs['layers'], kwargs['T'], kwargs['polarization'], kwargs['t'],
            **{k: kwargs[k] for k in kwargs if k in ['radius', 'a', 'b', 'straight_length']}
        )
    elif case == 'mrr_add_drop':
        return single_mrr_add_drop(
            wl, kwargs['resonator_type'], kwargs['layers'], kwargs['T'], kwargs['polarization'],
            kwargs['t1'], kwargs['kappa1'], kwargs.get('t2', None), kwargs.get('kappa2', None),
            **{k: kwargs[k] for k in kwargs if k in ['radius', 'a', 'b', 'straight_length']}
        )
    elif case == 'multi_mrr':
        return cascaded_mrrs_add_drop(wl, kwargs['params_list'], kwargs.get('bus_segments', None))
    else:
        raise ValueError(f"Unsupported case: {case}")

# -------------------------
# Visualization
# -------------------------
def plot_transmission(wl, T_thru, T_drop=None, labels=None):
    """Plot the transmission spectra.

    Parameters:
        wl (array): Wavelength array (m).
        T_thru (array): Through-port power transmission.
        T_drop (array or list, optional): Drop-port power transmission(s).
        labels (list, optional): Plot labels.

    Returns:
        None (displays plot).
    """
    plt.figure()
    plt.plot(wl * 1e9, T_thru, label=labels[0] if labels else 'Through')
    if T_drop is not None:
        if isinstance(T_drop, list):
            for i, T_d in enumerate(T_drop, 1):
                plt.plot(wl * 1e9, T_d, label=labels[i+1] if labels and i+1 < len(labels) else f'Drop {i}')
        else:
            plt.plot(wl * 1e9, T_drop, label=labels[1] if labels and len(labels) > 1 else 'Drop')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Power Transmission')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mode_profile(E_field, layers, T=25, labels=None, wl=None):
    """Plot waveguide mode profiles with refractive index overlay.

    Parameters:
        E_field (array): Mode profiles (shape: [num_modes, N_grid]).
        layers (list): List of dicts with material, thickness (m), n, dn_dT (1/°C).
        T (float): Temperature (°C, default=25).
        labels (list, optional): Plot labels.
        wl (float, optional): Wavelength for n calculation (m).

    Returns:
        None (displays plot).
    """
    total_thickness = sum(layer['thickness'] for layer in layers)
    N = E_field.shape[1]
    x = np.linspace(0, total_thickness, N)
    fig, ax1 = plt.subplots()
    for i in range(E_field.shape[0]):
        ax1.plot(x * 1e6, np.abs(E_field[i]), label=labels[i] if labels and i < len(labels) else f'Mode {i}')
    ax1.set_xlabel('Position (µm)')
    ax1.set_ylabel('Field Amplitude', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    # Plot refractive index
    ax2 = ax1.twinx()
    boundaries = np.cumsum([0] + [layer['thickness'] for layer in layers])
    n = np.zeros_like(x)
    for i in range(len(layers)):
        mask = (x >= boundaries[i]) & (x < boundaries[i+1])
        n_layer = layers[i]['n'] + layers[i]['dn_dT'] * (T - 25)
        n[mask] = n_layer
    ax2.plot(x * 1e6, n, 'k--', label='Refractive Index')
    ax2.set_ylabel('Refractive Index', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    # Plot layer boundaries
    for b in boundaries[1:-1]:
        ax1.axvline(b * 1e6, color='k', linestyle='--', alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.show()

# -------------------------
# Example Execution
# -------------------------
if __name__ == "__main__":
    # Use higher resolution to ensure stable wavelength spacing
    wl = np.linspace(1540e-9, 1560e-9, 1001)  # Increased points for stability
    
    # Define layers (silica/silicon/silica slab waveguide)
    layers = [
        {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': 10},
        {'material': 'silicon', 'thickness': 220e-9, 'n': 3.48, 'dn_dT': 1.8e-4, 'alpha': 100},
        {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': 10}
    ]
    
    # Example 1: Waveguide
    E_wg, T_wg = compute_transmission(
        case='waveguide', wavelengths=wl, layers=layers, T=30, polarization='TE', length=100e-6
    )
    plot_transmission(wl, T_wg, labels=['Waveguide'])
    
    # Example 2: Circular MRR with Polarization
    _, T_thru_TE = compute_transmission(
        case='mrr_thru', wavelengths=wl, resonator_type='circular', layers=layers,
        T=30, polarization='TE', t=0.9, radius=10e-6
    )
    _, T_thru_TM = compute_transmission(
        case='mrr_thru', wavelengths=wl, resonator_type='circular', layers=layers,
        T=30, polarization='TM', t=0.9, radius=10e-6
    )
    plot_transmission(wl, T_thru_TE, T_thru_TM, labels=['TE Polarization', 'TM Polarization'])
    
    # Example 3: Add-Drop Racetrack MRR with Complex Coupling
    def alpha_wl(wl):
        return 50 + 20 * (wl - 1540e-9) / (1560e-9 - 1540e-9)
    layers_racetrack = [
        {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': 10},
        {'material': 'silicon', 'thickness': 220e-9, 'n': 3.48, 'dn_dT': 1.8e-4, 'alpha': alpha_wl},
        {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': 10}
    ]
    kappa1 = np.sqrt(1 - 0.9**2) * np.exp(1j * np.pi/4)
    kappa2 = np.sqrt(1 - 0.85**2) * np.exp(1j * np.pi/6)
    (_, _), (T_thru_racetrack, T_drop_racetrack) = compute_transmission(
        case='mrr_add_drop', wavelengths=wl, resonator_type='racetrack', layers=layers_racetrack,
        T=30, polarization='TE', t1=0.9, kappa1=kappa1, t2=0.85, kappa2=kappa2,
        radius=10e-6, straight_length=5e-6
    )
    plot_transmission(wl, T_thru_racetrack, T_drop_racetrack, labels=['Racetrack Through', 'Racetrack Drop'])
    
    # Example 4: Cascaded MRRs with Bus Loss
    params_list = [
        {'resonator_type': 'circular', 'radius': 10e-6, 'layers': layers_racetrack,
         'T': 30, 'polarization': 'TE', 't1': 0.9, 'kappa1': np.sqrt(1 - 0.9**2)},
        {'resonator_type': 'elliptical', 'a': 12e-6, 'b': 8e-6, 'layers': layers,
         'T': 30, 'polarization': 'TE', 't1': 0.85, 'kappa1': np.sqrt(1 - 0.85**2)}
    ]
    bus_segments = [{'length': 50e-6, 'alpha': 10, 'n_eff': 1.5}]
    _, T_multi, drops = compute_transmission(
        case='multi_mrr', wavelengths=wl, params_list=params_list, bus_segments=bus_segments
    )
    plot_transmission(wl, T_multi, drops, labels=['Cascaded Through', 'Drop 1', 'Drop 2'])
    
    # Example 5: Mode Profile and Group Index
    E_field, n_eff = solve_waveguide_modes(wl, layers, T=30, polarization='TE', num_modes=2)
    plot_mode_profile(E_field[50], layers, T=30, labels=[f'Mode {i}' for i in range(2)], wl=wl[50])
    n_g = compute_group_index(wl, n_eff[:, 0])
    plt.figure()
    plt.plot(wl * 1e9, n_g, label='Group Index (Mode 0)')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Group Index')
    plt.legend()
    plt.grid(True)
    plt.show()
