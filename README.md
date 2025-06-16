# RingPDL: Photonic Ring Resonator Simulation Library

## Overview

**RingPDL** is a Python library for simulating optical transmission through photonic components, including straight waveguides, single microring resonators (MRRs), and cascaded MRR systems. It supports multiple resonator geometries (circular, elliptical, racetrack), polarization-dependent effects, wavelength-dependent loss, asymmetric coupling, bus waveguide loss, and multilayer material structures. A full-vectorial electromagnetic (EM) solver based on the finite-difference method (FDM) with perfectly matched layer (PML) boundary conditions computes accurate effective refractive indices (\( n_{\text{eff}} \)), group indices (\( n_g \)), and loss coefficients (\( \alpha \)) for multilayer waveguides. The library uses a transfer matrix approach with complex-field propagation to model phase, loss, and interference effects, supporting complex coupling coefficients for phase-sensitive devices.

### Key Features
- **Component Support**:
  - Straight waveguides (complex field and power transmission).
  - Single MRRs with through-port or add-drop configurations.
  - Cascaded MRRs with bus waveguide loss.
- **Geometry Flexibility**: Circular, elliptical, and racetrack resonators.
- **Electromagnetic Solver**: Full-vectorial FDM with PML for accurate mode propagation.
- **Material Layers**: Multilayer structures with material-specific properties.
- **Physical Accuracy**: Full round-trip loss, complex-field propagation, complex coupling coefficients.
- **Polarization Dependence**: TE and TM modes with vectorial solver.
- **Wavelength-Dependent Loss**: Constant, array-based, or callable loss coefficients.
- **Dispersion Analysis**: Group index calculation.
- **Visualization**: Transmission spectra and mode profiles with refractive index overlay.
- **Performance**: Caching for mode solutions and precomputed terms.

## Installation

Requires Python 3.6+ and:
- `numpy` for numerical computations.
- `scipy` for eigenvalue solving.
- `matplotlib` for plotting (optional).

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

Download `ringpdl.py` and place it in your project directory.

## Usage

The `compute_transmission` function selects simulations based on the `case` parameter. Examples demonstrate the vectorial solver, dispersion, and multilayer support.

### Example 1: Waveguide with Multilayer Structure
```python
import numpy as np
import ringpdl

wl = np.linspace(1540e-9, 1560e-9, 1000)  # Wavelengths (m)
layers = [
    {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': 10},
    {'material': 'silicon', 'thickness': 220e-9, 'n': 3.48, 'dn_dT': 1.8e-4, 'alpha': 100},
    {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': 10}
]
E_wg, T_wg = ringpdl.compute_transmission(
    case='waveguide',
    wavelengths=wl,
    layers=layers,
    T=30,
    polarization='TE',
    length=100e-6
)
ringpdl.plot_transmission(wl, T_wg, labels=['Waveguide'])
```

### Example 2: Add-Drop Elliptical MRR with Complex Coupling
```python
kappa1 = np.sqrt(1 - 0.9**2) * np.exp(1j * np.pi/4)  # Complex coupling
kappa2 = np.sqrt(1 - 0.85**2) * np.exp(1j * np.pi/6)
(_, _), (T_thru, T_drop) = ringpdl.compute_transmission(
    case='mrr_add_drop',
    wavelengths=wl,
    resonator_type='elliptical',
    layers=layers,
    T=30,
    polarization='TE',
    t1=0.9,
    kappa1=kappa1,
    t2=0.85,
    kappa2=kappa2,
    a=12e-6,
    b=8e-6
)
ringpdl.plot_transmission(wl, T_thru, T_drop, labels=['Elliptical Through', 'Elliptical Drop'])
```

### Example 3: Cascaded MRRs with Wavelength-Dependent Loss
```python
def alpha_wl(wl):
    return 50 + 20 * (wl - 1540e-9) / (1560e-9 - 1540e-9)

params_list = [
    {'resonator_type': 'circular', 'radius': 10e-6, 'layers': [
        {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': alpha_wl},
        {'material': 'silicon', 'thickness': 220e-9, 'n': 3.48, 'dn_dT': 1.8e-4, 'alpha': 100},
        {'material': 'silica', 'thickness': 2e-6, 'n': 1.45, 'dn_dT': 1e-5, 'alpha': 10}
    ], 'T': 30, 'polarization': 'TE', 't1': 0.9, 'kappa1': np.sqrt(1 - 0.9**2)},
    {'resonator_type': 'racetrack', 'radius': 10e-6, 'straight_length': 5e-6, 'layers': layers,
     'T': 30, 'polarization': 'TE', 't1': 0.85, 'kappa1': np.sqrt(1 - 0.85**2)}
]
bus_segments = [{'length': 50e-6, 'alpha': 10, 'n_eff': 1.5}]
_, T_multi, drops = ringpdl.compute_transmission(
    case='multi_mrr',
    wavelengths=wl,
    params_list=params_list,
    bus_segments=bus_segments
)
ringpdl.plot_transmission(wl, T_multi, drops, labels=['Cascaded Through', 'Drop 1', 'Drop 2'])
```

### Example 4: Mode Profile and Group Index
```python
E_field, n_eff = ringpdl.solve_waveguide_modes(wl, layers, T=30, polarization='TE', num_modes=2)
ringpdl.plot_mode_profile(E_field[500], layers, labels=['Mode 0', 'Mode 1'], wl=wl[500])
n_g = ringpdl.compute_group_index(wl, n_eff[:, 0])  # Group index for fundamental mode
plt.plot(wl * 1e9, n_g)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Group Index')
plt.show()
```

## API Reference

### Utility Functions

#### `circular_circumference(radius)`
- **Description**: Calculates the circumference of a circular ring.
- **Parameters**:
  - `radius`: Radius (m).
- **Returns**: Circumference (m).
- **Equation**: \( C = 2\pi r \)

#### `elliptical_circumference(a, b)`
- **Description**: Approximates the circumference of an elliptical ring using Ramanujan's formula.
- **Parameters**:
  - `a`: Semi-major axis (m).
  - `b`: Semi-minor axis (m).
- **Returns**: Circumference (m).
- **Equation**: \( C \approx \pi (a + b) \left(1 + \frac{3h}{10 + \sqrt{4 - 3h}}\right), h = \frac{(a-b)^2}{(a+b)^2} \)

#### `racetrack_circumference(radius, straight_length)`
- **Description**: Calculates the circumference of a racetrack resonator.
- **Parameters**:
  - `radius`: Radius of curved sections (m).
  - `straight_length`: Length of straight segments (m).
- **Returns**: Circumference (m).
- **Equation**: \( C = 2\pi r + 2L_s \)

#### `get_circumference(resonator_type, **geometry_params)`
- **Description**: Selects the appropriate circumference calculation.
- **Parameters**:
  - `resonator_type`: 'circular', 'elliptical', or 'racetrack'.
  - `geometry_params`: Parameters like `radius`, `a`, `b`, `straight_length` (m).
- **Returns**: Circumference (m).
- **Raises**: `ValueError` if inputs are invalid.

#### `solve_waveguide_modes(wl, layers, T, polarization='TE', num_modes=1, dx=10e-9)`
- **Description**: Solves the vector wave equation for waveguide modes using FDM with PML.
- **Parameters**:
  - `wl`: Wavelength (m, scalar or array).
  - `layers`: List of dicts with `material` (str), `thickness` (m), `n` (refractive index), `dn_dT` (1/°C), `alpha` (1/m).
  - `T`: Temperature (°C).
  - `polarization`: 'TE' or 'TM'.
  - `num_modes`: Number of modes to compute.
  - `dx`: Grid spacing (m, default=10 nm).
- **Returns**:
  - `E_field`: Mode profiles (shape: [len(wl), num_modes, N_grid] or [num_modes, N_grid] for scalar `wl`).
  - `n_eff`: Effective refractive indices (shape: [len(wl), num_modes] or [num_modes]).
- **Raises**: `ValueError` for invalid inputs.
- **Equation**: For TE (\( E_y \)):
  \[
  \frac{d^2 E_y}{dx^2} + \left( k_0^2 n^2(x) - \beta^2 \right) E_y = 0
  \]
  With PML: Complex coordinate stretching.

#### `compute_group_index(wl, n_eff)`
- **Description**: Calculates the group index from effective refractive index dispersion.
- **Parameters**:
  - `wl`: Wavelength array (m).
  - `n_eff`: Effective refractive index array (same shape as `wl`).
- **Returns**: Group index array.
- **Equation**: \( n_g = n_{\text{eff}} - \lambda \frac{d n_{\text{eff}}}{d \lambda} \)

#### `compute_layered_alpha_from_mode(layers, wl, E_field, dx)`
- **Description**: Computes effective loss coefficient based on mode confinement.
- **Parameters**:
  - `layers`: List of dicts with `material`, `thickness`, `alpha`.
  - `wl`: Wavelength array (m).
  - `E_field`: Mode profile (shape: [N_grid]).
  - `dx`: Grid spacing (m).
- **Returns**: Effective loss coefficient (1/m).
- **Raises**: `ValueError` for invalid inputs.
- **Equation**: \( \alpha_{\text{eff}} = \sum_i \alpha_i \frac{\int_{\text{layer } i} |E|^2 dx}{\int |E|^2 dx} \)

#### `propagation_phase(n_eff, wavelength, length)`
- **Description**: Computes the complex phase factor.
- **Parameters**:
  - `n_eff`: Effective refractive index.
  - `wavelength`: Wavelength (m).
  - `length`: Propagation length (m).
- **Returns**: Complex phase factor.
- **Raises**: `ValueError` if inputs are invalid.
- **Equation**: \( e^{-j \beta L}, \beta = \frac{2\pi n_{\text{eff}}}{\lambda} \)

#### `evaluate_alpha(alpha, wl)`
- **Description**: Evaluates the loss coefficient.
- **Parameters**:
  - `alpha`: Loss coefficient (1/m, scalar, array, or callable).
  - `wl`: Wavelength array (m).
- **Returns**: Loss coefficient array (1/m).
- **Raises**: `ValueError` if invalid.

### Component Functions

#### `single_waveguide(wl, layers, T, polarization, length)`
- **Description**: Computes complex field and power transmission for a waveguide.
- **Parameters**:
  - `wl`: Wavelength array (m).
  - `layers`: List of dicts with `material`, `thickness`, `n`, `dn_dT`, `alpha`.
  - `T`: Temperature (°C).
  - `polarization`: 'TE' or 'TM'.
  - `length`: Waveguide length (m).
- **Returns**:
  - `E`: Complex field transmission.
  - `T_power`: Power transmission.
- **Raises**: `ValueError` for invalid inputs.
- **Equation**: \( E = e^{-\alpha L / 2} e^{-j \beta L} \)

#### `single_mrr_thru(wl, resonator_type, layers, T, polarization, t, **geometry_params)`
- **Description**: Computes through-port transmission for an all-pass MRR.
- **Parameters**:
  - `wl`: Wavelength array (m).
  - `resonator_type`: 'circular', 'elliptical', or 'racetrack'.
  - `layers`: List of dicts with `material`, `thickness`, `n`, `dn_dT`, `alpha`.
  - `T`: Temperature (°C).
  - `polarization`: 'TE' or 'TM'.
  - `t`: Complex through-coupling coefficient.
  - `geometry_params`: Parameters like `radius`, `a`, `b`, `straight_length` (m).
- **Returns**:
  - `H`: Complex transfer function.
  - `T_power`: Power transmission.
- **Raises**: `ValueError` for invalid inputs.
- **Equation**: \( H = \frac{t - A e^{-j \phi}}{1 - t A e^{-j \phi}}, A = e^{-\alpha L}, \phi = \frac{2\pi n_{\text{eff}} L}{\lambda} \)

#### `single_mrr_add_drop(wl, resonator_type, layers, T, polarization, t1, kappa1, t2=None, kappa2=None, **geometry_params)`
- **Description**: Computes through and drop port transmissions for an add-drop MRR.
- **Parameters**:
  - `wl`: Wavelength array (m).
  - `resonator_type`: 'circular', 'elliptical', or 'racetrack'.
  - `layers`: List of dicts with `material`, `thickness`, `n`, `dn_dT`, `alpha`.
  - `T`: Temperature (°C).
  - `polarization`: 'TE' or 'TM'.
  - `t1`, `kappa1`: Complex input coupler coefficients (\( |t1|^2 + |\kappa1|^2 = 1 \)).
  - `t2`, `kappa2`: Complex output coupler coefficients (default to `t1`, `kappa1`).
  - `geometry_params`: Parameters like `radius`, `a`, `b`, `straight_length` (m).
- **Returns**:
  - `(H_thru, H_drop)`: Complex transfer functions.
  - `(T_thru, T_drop)`: Power transmissions.
- **Raises**: `ValueError` if coupling coefficients or inputs are invalid.
- **Equations**:
  \[
  H_{\text{thru}} = \frac{t_1 - t_2 A e^{-j \phi}}{1 - t_1 t_2 A e^{-j \phi}}, \quad
  H_{\text{drop}} = -\kappa_1 \kappa_2 \sqrt{A} e^{-j \phi / 2} / (1 - t_1 t_2 A e^{-j \phi})
  \]

#### `cascaded_mrrs_add_drop(wl, params_list, bus_segments=None)`
- **Description**: Computes transmission for cascaded add-drop MRRs.
- **Parameters**:
  - `wl`: Wavelength array (m).
  - `params_list`: List of dicts with `resonator_type`, `layers`, `T`, `polarization`, `t1`, `kappa1`, `t2`, `kappa2`, geometry parameters.
  - `bus_segments`: List of dicts with `length` (m), `alpha` (1/m), `n_eff` (length N-1 for N rings, optional).
- **Returns**:
  - `E_thru`: Final through-port complex field.
  - `T_thru`: Final through-port power.
  - `drop_powers`: List of drop-port power arrays.
- **Raises**: `ValueError` for invalid inputs.

### Selector Function

#### `compute_transmission(case, **kwargs)`
- **Description**: Selects and executes the simulation.
- **Parameters**:
  - `case`: 'waveguide', 'mrr_thru', 'mrr_add_drop', or 'multi_mrr'.
  - `kwargs`: Case-specific parameters.
- **Returns**: Outputs from the corresponding function.
- **Raises**: `ValueError` if invalid.

### Visualization

#### `plot_transmission(wl, T_thru, T_drop=None, labels=None)`
- **Description**: Plots transmission spectra.
- **Parameters**:
  - `wl`: Wavelength array (m).
  - `T_thru`: Through-port power transmission.
  - `T_drop`: Drop-port power transmission (single array or list, optional).
  - `labels`: List of plot labels.
- **Returns**: None (displays plot).

#### `plot_mode_profile(E_field, layers, labels=None, wl=None)`
- **Description**: Plots waveguide mode profiles with refractive index overlay.
- **Parameters**:
  - `E_field`: Mode profiles (shape: [num_modes, N_grid]).
  - `layers`: List of dicts with `material`, `thickness`, `n`, `dn_dT`.
  - `labels`: List of plot labels.
  - `wl`: Wavelength for refractive index calculation (m, optional).
- **Returns**: None (displays plot).

## Notes
- **Coordinate System**: \( x \)-axis (transverse, across layers), \( z \)-axis (propagation), \( y \)-axis (uniform for slab waveguides).
- **Units**: Lengths (m), loss coefficients (nepers/m), wavelengths (m), temperatures (°C).
- **EM Solver**: Vectorial FDM with PML, solving for TE (\( E_y \)) and TM (\( H_y \)) modes.
- **Polarization**: TE and TM modes handled via vectorial solver.
- **Coupling**: Complex coefficients supported for phase-sensitive devices.
- **Caching**: Mode solutions cached for performance.

## Limitations
- EM solver limited to slab waveguides; curved or 3D structures require advanced solvers.
- Computational cost increases with wavelength range and grid resolution.
- No direct ring-to-ring coupling (e.g., CROWs).

## Future Enhancements
- Bending loss calculation.
- Coupled-mode theory for perturbed resonators.
- Thermal crosstalk modeling.
- 3D or curved resonator solvers.

## License
MIT License.

## Contact
For issues or contributions, contact maintainers via GitHub or email.
