# RingsPDL: Rings-based CMT Toolkit

[Photonics Simulation]

**RingsPDL** is a comprehensive Python toolkit developed by the Photonic Device Lab at HKUST for simulating and analyzing photonic devices, with a special focus on ring resonator-based systems using Coupled Mode Theory (CMT).

**Author**: Elbek J. Keskinoglu (ejkeskinoglu@connect.ust.hk)

## Key Features

- **Advanced Ring Resonator Modeling**:
  - Circular, elliptical, and racetrack resonator geometries
  - Single and cascaded microring resonator systems
  - Add-drop and all-pass configurations
  - Complex coupling coefficient support

- **Waveguide Analysis**:
  - Finite-difference mode solver with PML boundaries
  - TE/TM polarization support
  - Loss calculation and group index computation
  - Temperature-dependent effects modeling

- **Material Database**:
  - Predefined materials (Silicon, Silica, Si3N4)
  - Sellmeier equations and tabulated refractive index data
  - Temperature-dependent refractive index models

- **Visualization Tools**:
  - Transmission spectra plotting
  - Mode profile visualization
  - Loss breakdown analysis
  - Temperature-dependent effects visualization

- **Performance Optimizations**:
  - LRU caching for mode solutions
  - Parallel computation support
  - Sparse matrix operations
  - Adaptive PML parameters

## Installation

```bash
pip install RingsPDL
```

## Dependencies

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- pytest (for tests)

## Quick Start

### Basic Ring Resonator Simulation

```python
import numpy as np
from RingsPDL import compute_transmission, plot_transmission

# Define wavelength range
wl = np.linspace(1540e-9, 1560e-9, 1000)

# Define waveguide structure
layers = [
    {'material': 'silica', 'thickness': 2e-6},
    {'material': 'silicon', 'thickness': 220e-9},
    {'material': 'silica', 'thickness': 2e-6}
]

# Calculate MRR transmission
_, T_thru = compute_transmission(
    case='mrr_thru',
    wavelengths=wl,
    resonator_type='circular',
    layers=layers,
    T=30,
    polarization='TE',
    t=0.9,
    radius=10e-6
)

# Plot results
plot_transmission(wl, T_thru, labels=['MRR Through Port'])
```

### Cascaded Ring Resonators

```python
# Define parameters for cascaded MRRs
params_list = [
    {
        'resonator_type': 'circular',
        'radius': 10e-6,
        'layers': layers,
        'T': 30,
        'polarization': 'TE',
        't1': 0.9,
        'kappa1': np.sqrt(1 - 0.9**2)
    },
    {
        'resonator_type': 'elliptical',
        'a': 12e-6,
        'b': 8e-6,
        'layers': layers,
        'T': 30,
        'polarization': 'TE',
        't1': 0.85,
        'kappa1': np.sqrt(1 - 0.85**2)
    }
]

bus_segments = [{'length': 50e-6, 'alpha': 10}]

# Calculate transmission
_, T_multi, drops = compute_transmission(
    case='multi_mrr',
    wavelengths=wl,
    params_list=params_list,
    bus_segments=bus_segments
)

# Plot results
plot_transmission(wl, T_multi, drops, 
                 labels=['Through Port', 'Drop Port 1', 'Drop Port 2'])
```

### Mode Profile Visualization

```python
from RingsPDL import solve_waveguide_modes, plot_mode_profile

# Solve for waveguide modes
E_field, n_eff = solve_waveguide_modes(
    wl, 
    layers, 
    T=30, 
    polarization='TE', 
    num_modes=2
)

# Plot mode profiles
plot_mode_profile(
    [E_field[0][500], 
    layers, 
    labels=['Fundamental Mode'],
    wl=wl[500]
)
```

## Advanced Features

### Temperature-Dependent Analysis

```python
temperatures = [25, 50, 75]
T_wg_temp = []

for T in temperatures:
    _, T_wg = compute_transmission(
        case='waveguide',
        wavelengths=wl,
        layers=layers,
        T=T,
        polarization='TE',
        length=100e-6
    )
    T_wg_temp.append(T_wg)

# Plot temperature-dependent transmission
import matplotlib.pyplot as plt
plt.figure()
for T, T_wg in zip(temperatures, T_wg_temp):
    plt.plot(wl * 1e9, T_wg, label=f'T = {T}°C')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.legend()
plt.grid(True)
plt.show()
```

### Complex Coupling Coefficients

```python
# Asymmetric MRR with complex coupling phases
kappa1_asym = np.sqrt(1 - 0.95**2) * np.exp(1j * np.pi/3)
kappa2_asym = np.sqrt(1 - 0.9**2) * np.exp(1j * 2*np.pi/3)

(_, _), (T_thru_asym, T_drop_asym) = compute_transmission(
    case='mrr_add_drop',
    wavelengths=wl,
    resonator_type='circular',
    layers=layers,
    T=30,
    polarization='TE',
    t1=0.95,
    kappa1=kappa1_asym,
    t2=0.9,
    kappa2=kappa2_asym,
    radius=10e-6
)

plot_transmission(wl, T_thru_asym, T_drop_asym,
                 labels=['Asymmetric MRR Through', 'Asymmetric MRR Drop'])
```

## Documentation

### Core Classes

1. **`LRUCache`**: Least Recently Used cache for storing mode solutions
2. **Material Database**: Predefined materials with optical properties
   - Access via `materials['silicon']['n'](wl, T)`

### Key Functions

- **Mode Solving**:
  - `solve_waveguide_modes`: Main mode solver
  - `solve_waveguide_modes_single_wl`: Single-wavelength mode solver
  - `compute_layered_alpha_from_mode`: Loss calculation
  - `compute_group_index`: Group index calculation

- **Resonator Analysis**:
  - `single_mrr_thru`: All-pass ring resonator
  - `single_mrr_add_drop`: Add-drop ring resonator
  - `cascaded_mrrs_add_drop`: Cascaded resonator systems
  - `get_circumference`: Geometry-specific circumference calculation

- **Utilities**:
  - `auto_pml_params`: Adaptive PML boundary calculation
  - `interpolate_complex`: Complex-valued interpolation
  - `propagation_phase`: Phase accumulation calculation
  - `quadratic_extrapolate`: Boundary condition extrapolation

### Visualization

- `plot_transmission`: Transmission spectra plotting
- `plot_mode_profile`: Mode field visualization

## Examples

The toolkit includes comprehensive examples demonstrating:

1. Basic waveguide transmission
2. Polarization effects in MRRs
3. Racetrack resonators with wavelength-dependent loss
4. Cascaded MRRs with bus loss
5. Temperature-dependent effects
6. Group index calculations
7. Asymmetric coupling in MRRs
8. Mode profile visualization
9. Heterogeneous cascaded MRRs

To run the examples, execute the script directly:

```bash
python RingsPDL.py
```

## Running Tests

Validate the implementation using the built-in test suite:

```bash
pytest RingsPDL.py
```

Tests include:
- Analytical waveguide verification
- Power conservation checks
- Group index calculation validation
- Spline interpolation accuracy
- Error handling for invalid inputs

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new feature branch
3. Implement your changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or feature requests, please contact:

**Elbek J. Keskinoglu**  
Email: [ejkeskinoglu@connect.ust.hk](mailto:ejkeskinoglu@connect.ust.hk)  
Photonic Device Lab, HKUST
