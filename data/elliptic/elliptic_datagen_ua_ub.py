'''
Generate training data ua and ub for nonsmooth optimal control of elliptic PDEs
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
npz_filename = (
    "elliptic_ua_ub_test_res64_sz2048.npz"  # filename for saving the generated data
)

ua_range = [-10.0, -1.0]
ub_range = [1.0, 10.0]
alpha = 4
tau = 3

k1, k2 = np.meshgrid(range(s + 1), range(s + 1))
coef = np.sqrt(2.0) * (4.0 * np.pi**2 * (k1**2 + k2**2) + tau**2) ** (-0.5 * alpha)
ua = np.zeros((n, m + 1, m + 1))
ub = np.zeros((n, m + 1, m + 1))


# Generate data u_a

print("Generating u_a...")

for i in tqdm(range(n), total=n, bar_format="{l_bar}{bar:36}{r_bar}"):

    xi = np.random.standard_normal((s + 1, s + 1))
    L = s**2 * coef * xi
    L[0, 0] = 0
    ua_fine = (idstn(L, norm="ortho") + idctn(L, norm="ortho")) / np.sqrt(2.0)
    ua_fine = ua_fine + np.random.uniform(ua_range[0], ua_range[1]) * np.ones(
        (s + 1, s + 1)
    )
    ua_fine = np.minimum(ua_fine, -1e-6)

    ua[i, :, :] = ua_fine[::4, ::4]  # type: ignore


# Generate data u_b

print("Generating u_b...")

for i in tqdm(range(n), total=n, bar_format="{l_bar}{bar:36}{r_bar}"):

    xi = np.random.standard_normal((s + 1, s + 1))
    L = s**2 * coef * xi
    L[0, 0] = 0
    ub_fine = (idstn(L, norm="ortho") + idctn(L, norm="ortho")) / np.sqrt(2.0)
    ub_fine = ub_fine + np.random.uniform(ub_range[0], ub_range[1]) * np.ones(
        (s + 1, s + 1)
    )
    ub_fine = np.maximum(ub_fine, 1e-6)

    ub[i, :, :] = ub_fine[::4, ::4]  # type: ignore


# Save generated data

with open(npz_filename, "wb") as file:
    np.savez(file, ua=ua, ub=ub)


# Test and plot generated data

if debug:
    with open(npz_filename, "rb") as file:
        data = np.load(file)
        ua_read, ub_read = data["ua"], data["ub"]
    print(ua_read.shape)
    print(ub_read.shape)

    x, y = np.meshgrid(range(m + 1), range(m + 1))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(     # type: ignore
        x,
        y,
        ua[0, :, :],
        cmap=cm.coolwarm,  # type: ignore
        linewidth=0,
        antialiased=False,
    )
    plt.savefig('elliptic_ua_0.pdf')
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(     # type: ignore
        x,
        y,
        ub[0, :, :],
        cmap=cm.coolwarm,  # type: ignore
        linewidth=0,
        antialiased=False,
    )
    plt.savefig('elliptic_ub_0.pdf')
    plt.show()
