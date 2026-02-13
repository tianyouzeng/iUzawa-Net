'''
Generate training data for optimal control of parabolic PDEs
This script deals with source term and the desired state
We impose Dirichlet boundary conditions on the yd
'''

import numpy as np
from scipy.fft import idstn, idctn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from tqdm import tqdm

import os, sys, inspect
import multiprocessing

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

write_filename = os.path.join(
    root_dir, 'data/parabolic/parabolic_f_yd_test_res32_sz2048.npz'
)
debug = False

m = 32  # resolution for down-sampling, grid_size = m+1
s = 256  # resolution on Fourier space, this is 4*m, grid_size = s+1
n = 2048  # dataset size
alpha = 3
tau = 3

k1, k2, k3 = np.meshgrid(range(s + 1), range(s + 1), range(s + 1))
coef = (4.0 * np.pi**2 * (k1**2 + k2**2 + k3**2) + tau**2) ** (-0.5 * alpha)
f = np.zeros((n, m + 1, m + 1, m + 1))
yd = np.zeros((n, m + 1, m + 1, m + 1))


def init_pool_processes():
    np.random.seed()


''' Generate f '''

print("Generating f...")


def generate_f(idx: int):
    xi = np.random.standard_normal((s + 1, s + 1, s + 1))
    L = 0.1 * s**3 * coef * xi
    L[0, 0, 0] = 0
    f_fine = (idstn(L, norm="ortho") + idctn(L, norm="ortho")) / np.sqrt(2.0)
    return f_fine[::8, ::8, ::8]  # type: ignore


with multiprocessing.Pool(processes=64, initializer=init_pool_processes) as pool:
    results = list(
        tqdm(
            pool.imap(generate_f, range(n)),
            total=n,
            bar_format="{l_bar}{bar:36}{r_bar}",
        )
    )
    for i, f_i in enumerate(results):
        f[i] = f_i
    del results


''' Generate yd '''

print("Generating yd...")


def generate_yd(idx: int):
    xi = np.random.standard_normal((s + 1, s + 1, s + 1))
    L = 0.05 * s**3 * coef * xi
    L[0, 0, 0] = 0
    yd_fine = (idstn(L, norm="ortho") + idctn(L, norm="ortho")) / np.sqrt(2.0)
    return yd_fine[::8, ::8, ::8]  # type: ignore


with multiprocessing.Pool(processes=64, initializer=init_pool_processes) as pool:
    results = list(
        tqdm(
            pool.imap(generate_yd, range(n)),
            total=n,
            bar_format="{l_bar}{bar:36}{r_bar}",
        )
    )
    for i, yd_i in enumerate(results):
        yd[i] = yd_i
    del results


''' Save generated data '''

with open(write_filename, "wb") as file:
    np.savez(file, f=f, yd=yd)


''' Test and plot generated data '''

if debug:

    with open(write_filename, "rb") as file:
        data = np.load(file)
        f_read, yd_read = data["f"], data["yd"]
    print(f_read.shape)
    print(yd_read.shape)

    x, y = np.meshgrid(range(m + 1), range(m + 1))

    def update_surface(frame, array, ax):
        ax.clear()
        surf = ax.plot_surface(
            x,
            y,
            array[0, :, :, frame],
            cmap=cm.coolwarm,  # type: ignore
            linewidth=0,
            antialiased=False,
        )
        return (surf,)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ani = animation.FuncAnimation(
        fig, update_surface, fargs=(f_read, ax), frames=65, interval=50
    )
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ani = animation.FuncAnimation(
        fig, update_surface, fargs=(yd_read, ax), frames=65, interval=50
    )
    plt.show()
