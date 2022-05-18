# Quantum Impurity Relaxometry

Magnons are quanta of spin waves, which are modes of collectively precessing spins. Thermally excited magnons in thin magnetic films generate stray fields at the film surface which can be detected using quantum impurities such as nitrogen-vacancy (NV) centers. NVs are lattice defects in diamond and are able to couple with magnon stray fields. Assuming a thermal occupancy of magnon modes, we study the magnetization dynamics of the magnons propagating through thin magnetic insulators using the Landau-Lifshitz-Gilbert equation.

We implement a numerical model to predict and understand the response of the NV center to proximal magnons in thin films. The simulation includes a static bias field in an arbitrary orientation with respect to the quantization axis of the NV center using the diamond’s tetrahedral symmetry. This extended model is in demand due to limitations in present-day measurement techniques to align the bias field with an NV center. The code in this module is based on and an extension of the QIR theory presented in Rustagi et al. (2020) [1].

## Installation

Run the following to install this package:

```bash
pip install qir
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/helenala/qi-relaxometry/blob/main/notebooks/demo.ipynb)

## Usage

Import the Quantum Impurity Relaxometry (`qir`) module.

```python
from qir import RelaxationRate, ZFS, GAMMA
```

Import `numpy` required for matrix calculations.

```python
import numpy as np
from numpy import pi, linspace, empty_like
```

Import `matplotlib` required for plotting.

```python
import matplotlib.pyplot as plt
import matplotlib.colors as colors
```

Call `RelaxationRate` class and choose parameters.

```python
B_ext = 31e-3
Gamma = RelaxationRate(bext=B_ext,
                       quadrants="all",
                       zoom_in_heatmap=1.5,
                       film_thickness=235e-9)
```

Check the input arguments.

```python
Gamma.init_locals
```

Get kx, ky meshgrids. Create integrand meshgrid corresponding to kx and ky.

- If creating high-res 2D meshgrid plots, use at least `x_pixels = 500` and `y_pixels = 500`.
- If calculating rate `Gamma` vs field `B_ext`, use `x_pixels = 200` and `y_pixels = 6000`.

```python
Gamma.create_k_bounds()
Gamma.create_k_meshgrids(x_pixels=1000, y_pixels=1000)
Gamma.calculate_sum_di_dj_cij()
Gamma.create_integrand_grid_exclude_nv_distance()
Gamma.create_integrand_grid_include_nv_distance()
```

Get `k_x`, `k_y` and `Gamma` integrand.

```python
X = Gamma.kx * 1e-6 / (2*pi)  # 2D numpy array [1/um]
Y = Gamma.ky * 1e-6 / (2*pi)  # 2D numpy array [1/um]
Z = Gamma.integrand_grid_include_nv_distance  # 2D numpy array [rad Hz]
```

Calculate minimum and maximum element in `Gamma` integrand array.

```python
vmin = np.amin(Z)
vmax = np.amax(Z)
```

Create a directory where you can save your plots in.

```python
path = "plot-figures"

import os

# create directory in current folder
os.mkdir(path) if not os.path.exists(path) else None

path_txt = os.path.join(path, "txt_files")
path_fig = os.path.join(path, "figures")
os.mkdir(path_txt) if not os.path.exists(path_txt) else None
os.mkdir(path_fig) if not os.path.exists(path_fig) else None

print(f"Created: \n\t.{path_txt}")
print(f"Created: \n\t.{path_fig}")
```

Save meshgrids in `.txt` files (optional).

```python
B = B_ext * 1e4  # external magnetic field [Gauss]
path_txt_X = os.path.join(path, f"heatmap_{B:.0f}G_X.txt")
path_txt_Y = os.path.join(path, f"heatmap_{B:.0f}G_Y.txt")
path_txt_Z = os.path.join(path, f"heatmap_{B:.0f}G_Z.txt")
np.savetxt(path_txt_X, X, header=f"Meshgrid kx [1/um]; B_ext = {B:.0f} Gauss")
np.savetxt(path_txt_Y, Y, header=f"Meshgrid ky [1/um]; B_ext = {B:.0f} Gauss")
np.savetxt(path_txt_Z, Z, header=f"Log of Gamma integrand [Hz]; B_ext {B:.0f} Gauss")
print(f"Saved in: \n\t{path_txt_X}\n\t{path_txt_Y}\n\t{path_txt_Z}")
```

Plot 2D heatmap of `Gamma` integrand using `pyplot.pcolormesh` in `log_10` scale.

```python
fig1, ax1 = plt.subplots(figsize=(18,8))

ax1.set_xlabel(r"$k_x/2\mathregular{\pi}$ (1/$\mathregular{\mu}$m)", fontsize=18)
ax1.set_ylabel(r"$k_y/2\mathregular{\pi}$ (1/$\mathregular{\mu}$m)", fontsize=18)
im1 = ax1.pcolormesh(X, Y, Z, cmap='jet', norm=colors.LogNorm(vmin=1e-11, vmax=vmax), shading="auto")
cbar1 = fig1.colorbar(im1)
cbar1.set_label(r"$\Delta k \log_{10}(\sum_{i,j} D_i D_j C_{ij})$", fontsize=18)
```

Save plot.

```python
path_to_file = os.path.join(path_fig, "heatmap_logscale.png")
fig1.savefig(path_to_file)
print(f"Plot saved in: \n\t{path_to_file}")
```

Plot 2D heatmap `Gamma` integrand using `pyplot.pcolormesh` in normal scale.

```python
fig2, ax2 = plt.subplots(figsize=(18,8))
ax2.set_xlabel(r"$k_x/2\mathregular{\pi}$ (1/$\mathregular{\mu}$m)", fontsize=18)
ax2.set_ylabel(r"$k_y/2\mathregular{\pi}$ (1/$\mathregular{\mu}$m)", fontsize=18)
im2 = ax2.pcolormesh(X, Y, Z, cmap='jet', shading="auto")
cbar2 = fig2.colorbar(im2)
cbar2.set_label(r"$\Delta k \log_{10}(\sum_{i,j} D_i D_j C_{ij})$", fontsize=18)
```

Save plot.

```python
path_to_file = os.path.join(path_fig, "heatmap_logscale.png")
fig2.savefig(path_to_file)
print(f"Plot saved in: \n\t{path_to_file}")
```

### Relaxation rate as function of magnetic field

Calculate the relaxation rate `Gamma` [MHz] as function of the magnetic field `B` [Tesla].

```python
def get_rate_vs_field():
    x = linspace(1e-3, 50e-3, 25)
    y = empty_like(x)
    for i in range(len(x)):
        rate = RelaxationRate(A_exchange=8.47e-12,
                              distance_nv=40e-9,
                              Gilbert_damping=50e-3,
                              film_thickness=40e-9,
                              M_saturation=324000,
                              bext=x[i])
        rate.kx_min = -5e6 * 2*pi
        rate.kx_max = 5e6 * 2*pi
        rate.ky_min = -2e6 * 2*pi
        rate.ky_max = 2e6 * 2*pi
        rate.create_k_meshgrids(x_pixels=200, y_pixels=4000)
        rate.calculate_sum_di_dj_cij()
        rate.create_integrand_grid_exclude_nv_distance()
        rate.create_integrand_grid_include_nv_distance()
        y[i] = rate.rate_in_MHz
    return x, y
```

Plot `Gamma` as function of `B`.

```python
B, rate = get_rate_vs_field()

fig3, ax3 = plt.subplots(figsize=(8,6))
plt.plot(B, rate)
plt.xlabel("B (Tesla)")
plt.ylabel("Rate (MHz)")
plt.show()
```

### QI distance dependency

Calculate relaxation rate `Gamma` [MHz] as function of the distance `d` [m] between the quantum impurity (e.g. nitrogen-vacancy spin).

```python
def get_rate_vs_distance(B = 25e-3):
    x = np.geomspace(50e-9, 500e-9, 100)
    y = np.empty_like(x)
    rate = RelaxationRate(bext=B)
    rate.create_k_bounds()
    rate.create_k_meshgrids(x_pixels=100, y_pixels=4000)
    rate.calculate_sum_di_dj_cij()
    rate.create_integrand_grid_exclude_nv_distance()
    for i in range(len(x)):
        rate.distance_nv = x[i]
        rate.create_integrand_grid_include_nv_distance()
        y[i] = rate.rate_in_MHz
    return x, y
```

Plot `Gamma` [MHz] as function of distance `d` [meter].

```python
d, r = get_rate_vs_distance()
plt.plot(d, r)
plt.xlabel("Distance (m)")
plt.ylabel("Rate (MHz)")
plt.show()
```

### Spin-Wave Dispersion: DESWs and BVSWs

Calculate the relaxation rate for film thickness 235 nm.

```python
Gamma1 = RelaxationRate(film_thickness=235e-9)
```

Get the wave numbers from `k=0` to `k=8*2\pi` [1/um].

```python
k = np.linspace(0, 8e6 * 2 * pi, 10000)
```

Calculate the Backward-Volume Spin Wave Dispersion (BVSW).

```python
# BVSW (ky=0)
Gamma1.kx = k
Gamma1.ky = np.zeros_like(k)
Gamma1.create_w_meshgrids()
omega_bvsw = Gamma1.omega_spin_wave_dispersion()
```

Calculate the Damon-Eschbach Spin Wave Dispersion (BVSW).

```python
# DESW (kx=0)
Gamma1.ky = k
Gamma1.kx = np.zeros_like(k)
Gamma1.create_w_meshgrids()
omega_desw = Gamma1.omega_spin_wave_dispersion()
```

Plot the frequency `f` [GHz] as function of wave number `k` [1/um].

```python
plt.figure(figsize=(8,6))
plt.plot(k*1e-6/(2*pi), omega_desw*1e-9/(2*pi), label="DESW", c='b')
plt.plot(k*1e-6/(2*pi), omega_bvsw*1e-9/(2*pi), label="BVSW", c='r')
plt.xlabel("k (1/um)")
plt.ylabel("f (GHz)")
plt.legend()
```

## Authors

Developers:

- [Helena La](https://github.com/helenala) (theory + code)

Contributors:

- [Brecht Simon](https://github.com/brechtsimon) (theory + code testing)
- [Dr. Toeno van der Sar](https://www.tudelft.nl/tnw/over-faculteit/afdelingen/quantum-nanoscience/van-der-sar-lab) (theory)
- [Dr. Samer Kurdi](https://github.com/spectecals) (theory)

## Bibliography

[1] A. Rustagi, I. Bertelli, T. van der Sar, and P. Upadhyaya, 2020, https://doi.org/10.1103/PhysRevB.102.220403
