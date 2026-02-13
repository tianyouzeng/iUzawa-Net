'''
Generate training data for optimal control of elliptic PDEs
We impose Dirichlet boundary conditions on the yd
'''

import numpy as np
from scipy.fft import idstn, idctn
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

debug = False

m = 64  # resolution for down-sampling, grid_size = m+1
s = 256  # resolution on Fourier space, this is 4*m, grid_size = s+1
n = 2048  # dataset size
alpha = 3
tau = 3

k1, k2 = np.meshgrid(range(s + 1), range(s + 1))
coef = np.sqrt(2.0) * (4.0 * np.pi**2 * (k1**2 + k2**2) + tau**2) ** (-0.5 * alpha)
f = np.zeros((n, m + 1, m + 1))
yd = np.zeros((n, m + 1, m + 1))

# Generate data f

print("Generating f...")

for i in tqdm(range(n), total=n, bar_format="{l_bar}{bar:36}{r_bar}"):

    xi = np.random.standard_normal((s + 1, s + 1))
    L = s**2 * coef * xi
    L[0, 0] = 0
    f_fine = (idstn(L, norm="ortho") + idctn(L, norm="ortho")) / np.sqrt(2.0)
    f[i, :, :] = f_fine[::4, ::4]  # type: ignore

# Generate data y

print("Generating yd...")

for i in tqdm(range(n), total=n, bar_format="{l_bar}{bar:36}{r_bar}"):

    xi = np.random.standard_normal((s + 1, s + 1))
    L = s**2 * coef * xi
    L[0, 0] = 0
    yd_fine = idstn(L, norm="ortho")
    yd[i, :, :] = yd_fine[::4, ::4]  # type: ignore

# Save generated data

npz_filename = "elliptic_f_yd_zero_bd_test_res64_sz2048.npz"
with open(npz_filename, "wb") as file:
    np.savez(file, f=f, yd=yd)


# Test and plot generated data

if debug:

    with open(npz_filename, "rb") as file:
        data = np.load(file)
        f_read, yd_read = data["f"], data["yd"]
    print(f_read.shape)
    print(yd_read.shape)

    x, y = np.meshgrid(range(m + 1), range(m + 1))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(     # type: ignore
        x,
        y,
        f_read[0, :, :],
        cmap=cm.coolwarm,  # type: ignore
        linewidth=0,
        antialiased=False,
    )
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(     # type: ignore
        x,
        y,
        yd_read[0, :, :],
        cmap=cm.coolwarm,  # type: ignore
        linewidth=0,
        antialiased=False,
    )
    plt.show()
