'''
Generate training data for optimal control of elliptic PDEs
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
    root_dir, 'data/parabolic/parabolic_ua_ub_train_res32_sz2048.npz'
)
debug = True

m = 32  # resolution for down-sampling, grid_size = m+1
s = 256  # resolution on Fourier space, this is 4*m, grid_size = s+1
n = 2048  # dataset size

ua_range = [-5.0, -1.0]
ub_range = [1.0, 5.0]
alpha = 3
tau = 3

k1, k2, k3 = np.meshgrid(range(s + 1), range(s + 1), range(s + 1))
coef = (4.0 * np.pi**2 * (k1**2 + k2**2 + k3**2) + tau**2) ** (-0.5 * alpha)
ua = np.zeros((n, m + 1, m + 1, m + 1))
ub = np.zeros((n, m + 1, m + 1, m + 1))


def init_pool_processes():
    np.random.seed()


''' Generate ua '''

print("Generating ua...")


def generate_ua(idx: int):
    xi = np.random.standard_normal((s + 1, s + 1, s + 1))
    L = 0.01 * s**3 * coef * xi
    L[0, 0, 0] = 0
    ua_fine = (idstn(L, norm="ortho") + idctn(L, norm="ortho")) / np.sqrt(2.0)
    ua_fine = ua_fine + np.random.uniform(ua_range[0], ua_range[1]) * np.ones(
        (s + 1, s + 1, s + 1)
    )
    ua_fine = np.minimum(ua_fine, -1e-6)
    return ua_fine[::4, ::4, ::4]  # type: ignore


with multiprocessing.Pool(processes=20, initializer=init_pool_processes) as pool:
    results = list(
        tqdm(
            pool.imap(generate_ua, range(n)),
            total=n,
            bar_format="{l_bar}{bar:36}{r_bar}",
        )
    )
    for i, ua_i in enumerate(results):
        ua[i] = ua_i
    del results


''' Generate ub '''

print("Generating ub...")


def generate_ub(idx: int):
    xi = np.random.standard_normal((s + 1, s + 1, s + 1))
    L = 0.01 * s**3 * coef * xi
    L[0, 0, 0] = 0
    ub_fine = (idstn(L, norm="ortho") + idctn(L, norm="ortho")) / np.sqrt(2.0)
    ub_fine = ub_fine + np.random.uniform(ub_range[0], ub_range[1]) * np.ones(
        (s + 1, s + 1, s + 1)
    )
    ub_fine = np.maximum(ub_fine, 1e-6)
    return ub_fine[::4, ::4, ::4]  # type: ignore


import multiprocessing

with multiprocessing.Pool(processes=20, initializer=init_pool_processes) as pool:
    results = list(
        tqdm(
            pool.imap(generate_ub, range(n)),
            total=n,
            bar_format="{l_bar}{bar:36}{r_bar}",
        )
    )
    for i, ub_i in enumerate(results):
        ub[i] = ub_i
    del results


''' Save generated data '''

with open(write_filename, "wb") as file:
    np.savez(file, ua=ua, ub=ub)


''' Test and plot generated data '''

if debug:

    with open(write_filename, "rb") as file:
        data = np.load(file)
        ua_read, ub_read = data["ua"], data["ub"]
    print(ua_read.shape)
    print(ub_read.shape)

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
        fig, update_surface, fargs=(ua_read, ax), frames=65, interval=50, blit=False
    )
    plt.show()
    ani.save('ua_debug.gif', writer='pillow')

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ani = animation.FuncAnimation(
        fig, update_surface, fargs=(ub_read, ax), frames=65, interval=50, blit=False
    )
    plt.show()
    ani.save('ub_debug.gif', writer='pillow')

    def update_surface_1(frame, array, ax):
        ax.clear()
        surf = ax.plot_surface(
            x,
            y,
            array[1, :, :, frame],
            cmap=cm.coolwarm,  # type: ignore
            linewidth=0,
            antialiased=False,
        )
        return (surf,)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ani = animation.FuncAnimation(
        fig, update_surface_1, fargs=(ua_read, ax), frames=65, interval=50, blit=False
    )
    ani.save('ua_debug_1.gif', writer='pillow')
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ani = animation.FuncAnimation(
        fig, update_surface_1, fargs=(ub_read, ax), frames=65, interval=50, blit=False
    )
    ani.save('ub_debug_1.gif', writer='pillow')
    plt.show()
