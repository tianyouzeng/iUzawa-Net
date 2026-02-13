'''
Generate training data for optimal control of elliptic PDEs
Optimal control and state variables
Solve by semismooth Newton method with FEM discretization
'''

''' Import '''

import numpy as np
import scipy.sparse
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pypardiso

import multiprocessing
import functools
from timeit import default_timer
from tqdm import tqdm

import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)
from utils.fem_mesh import rectangleMesh, quadpts, TriMesh2D


''' Problem parameters '''

alpha = 0.01


''' Configs '''

data_size = -1  # -1 means use all data
max_iter_ssn = 20
use_multiprocessing = True
process_count = 4
debug = False

read_filename_f_yd = os.path.join(
    root_dir, 'data/elliptic/elliptic_f_yd_test_res64_sz2048.npz'
)
read_filename_ua_ub = os.path.join(
    root_dir, 'data/elliptic/elliptic_ua_ub_test_res64_sz2048.npz'
)
write_filename = os.path.join(
    root_dir, 'data/elliptic/elliptic_u_y_test_res64_sz2048.npz'
)


def boundary_cond_func(pos: npt.NDArray) -> npt.NDArray:
    '''Dirichlet boundary condition'''
    x1 = pos[:, 0]
    return 0 * x1


''' Space geometry '''

x_range = (0.0, 1.0)
y_range = (0.0, 1.0)
resol_space = 64
h = 1 / resol_space
node, elem = rectangleMesh(x_range, y_range, h)
tri_mesh = TriMesh2D(node, elem)
tri_mesh.update_auxstructure()
tri_mesh.update_gradbasis()
is_bd_node = tri_mesh.isBdNode
der_phi = tri_mesh.Dlambda
area: npt.NDArray = tri_mesh.area
phi, weight = quadpts()
num_quad = len(phi)
num_node = num_dof = len(node)
num_tri = len(elem)
free_node = ~is_bd_node
free_node_idx_list = free_node * np.cumsum(free_node) - 1
num_free_node = np.sum(free_node)


''' FEM matrices '''

# For most of the cases, only store the free_node submatrix is necessary for computation

# Stiff matrix
matrix_k = scipy.sparse.csc_matrix((num_free_node, num_free_node))
for i in range(3):
    for j in range(3):
        matrix_k_ij = area * (der_phi[..., i] * der_phi[..., j]).sum(axis=-1)
        coords_i_temp = free_node_idx_list[elem[:, i]]
        coords_j_temp = free_node_idx_list[elem[:, j]]
        ij_is_free_node = (coords_i_temp != -1) & (coords_j_temp != -1)
        coords_i = coords_i_temp[ij_is_free_node]
        coords_j = coords_j_temp[ij_is_free_node]
        matrix_k += scipy.sparse.csc_matrix(
            (matrix_k_ij[ij_is_free_node], (coords_i, coords_j)),
            shape=(num_free_node, num_free_node),
        )

# Mass matrix for the right hand side
matrix_m = scipy.sparse.csc_matrix((num_free_node, num_free_node))
for i in range(3):
    for j in range(3):
        matrix_m_ij = (1.0 + (i == j)) / 12.0 * area
        coords_i_temp = free_node_idx_list[elem[:, i]]
        coords_j_temp = free_node_idx_list[elem[:, j]]
        ij_is_free_node = (coords_i_temp != -1) & (coords_j_temp != -1)
        coords_i = coords_i_temp[ij_is_free_node]
        coords_j = coords_j_temp[ij_is_free_node]
        matrix_m += scipy.sparse.csc_matrix(
            (matrix_m_ij[ij_is_free_node], (coords_i, coords_j)),
            shape=(num_free_node, num_free_node),
        )

# Mass matrix (full), for computing inner product
matrix_m_full = scipy.sparse.csc_matrix((num_node, num_node))
for i in range(3):
    for j in range(3):
        matrix_m_ij = (1.0 + (i == j)) / 12.0 * area
        matrix_m_full += scipy.sparse.csc_matrix(
            (matrix_m_ij, (elem[:, i], elem[:, j])), shape=(num_dof, num_dof)
        )


def inner_prod(u: npt.NDArray, v: npt.NDArray) -> float:
    '''Compute the inner product of two functions u and v'''

    mu = matrix_m_full.dot(u)
    vmu: npt.NDArray = np.multiply(mu, v)
    return vmu.sum().item()


''' SSN Iterations '''


def psi_func(
    t: npt.NDArray, alpha: float, ua: npt.NDArray, ub: npt.NDArray
) -> npt.NDArray:
    '''Scalar valued function in the superposition operator'''
    prod = -(1.0 / alpha) * t
    psi = prod.clip(ua, ub)
    return psi


def init_ssn(yd: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    p = np.zeros(num_node)
    y = np.zeros(num_node)
    u = np.zeros(num_node)
    # u[free_node] = matrix_k * y[free_node]
    return y, u, p


def get_ssn_step(
    y: npt.NDArray,
    u: npt.NDArray,
    p: npt.NDArray,
    f: npt.NDArray,
    yd: npt.NDArray,
    alpha: float,
    ua: npt.NDArray,
    ub: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

    alpha_divides_p = (-1.0 / alpha) * p[free_node]
    is_inactive_nodes = (alpha_divides_p < ub[free_node]) * (
        alpha_divides_p > ua[free_node]
    )
    matrix_a = scipy.sparse.vstack(
        (
            scipy.sparse.hstack((matrix_m, -matrix_k)),
            scipy.sparse.hstack(
                (
                    -matrix_k,
                    (-1.0 / alpha)
                    * matrix_m
                    @ scipy.sparse.diags(is_inactive_nodes.astype(float)),
                )
            ),
        )
    )
    rhs = np.hstack(
        (
            matrix_m @ (y[free_node] - yd[free_node]) - matrix_k @ p[free_node],
            -matrix_k @ y[free_node]
            + matrix_m
            @ (
                psi_func(p[free_node], alpha, ua[free_node], ub[free_node])
                + f[free_node]
            ),
        )
    )
    y_p_step = pypardiso.spsolve(matrix_a, rhs)
    y_step = y_p_step[:num_free_node]
    p_step = y_p_step[num_free_node:]
    u_step = -(1.0 / alpha) * is_inactive_nodes * y_step + (
        u[free_node] - psi_func(p[free_node], alpha, ua[free_node], ub[free_node])
    )
    return y_step, u_step, p_step


def ssn(
    f: npt.NDArray,
    yd: npt.NDArray,
    alpha: float,
    ua: npt.NDArray,
    ub: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    '''SSN applied on single instance of f, yd'''
    f = f.reshape(-1)
    yd = yd.reshape(-1)
    ua = ua.reshape(-1)
    ub = ub.reshape(-1)
    y, u, p = init_ssn(yd)

    for _ in range(max_iter_ssn):
        y_step, u_step, p_step = get_ssn_step(y, u, p, f, yd, alpha, ua, ub)
        y[free_node] = y[free_node] - y_step
        u[free_node] = u[free_node] - u_step
        p[free_node] = p[free_node] - p_step

    u = u.reshape((resol_space + 1, resol_space + 1))
    y = y.reshape((resol_space + 1, resol_space + 1))
    p = p.reshape((resol_space + 1, resol_space + 1))
    return y, u, p


def ssn_star(
    f_yd_ua_ub_tuple: tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray],
    alpha: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return ssn(
        f_yd_ua_ub_tuple[0],
        f_yd_ua_ub_tuple[1],
        alpha,
        f_yd_ua_ub_tuple[2],
        f_yd_ua_ub_tuple[3],
    )


''' Generate optimal control and state variables '''

if __name__ == '__main__':

    with open(read_filename_f_yd, 'rb') as file:
        data = np.load(file)
        f_list, yd_list = data['f'], data['yd']
        data.close()
    assert f_list.shape[0] == yd_list.shape[0]
    if data_size == -1:
        data_size: int = f_list.shape[0]

    with open(read_filename_ua_ub, 'rb') as file:
        data = np.load(file)
        ua_list, ub_list = data['ua'], data['ub']
        data.close()
    assert ua_list.shape[0] == ub_list.shape[0] == f_list.shape[0]

    u_list = np.zeros((data_size, resol_space + 1, resol_space + 1))
    y_list = np.zeros((data_size, resol_space + 1, resol_space + 1))
    pool_map_func = functools.partial(
        ssn_star,
        alpha=alpha,
    )

    time_start = default_timer()

    if use_multiprocessing:
        pool_inputs = [
            (f_list[i, :, :], yd_list[i, :, :], ua_list[i, :, :], ub_list[i, :, :])
            for i in range(data_size)
        ]
        with multiprocessing.Pool(processes=process_count) as pool:
            results = list(
                tqdm(
                    pool.imap(pool_map_func, pool_inputs),
                    total=data_size,
                    bar_format="{l_bar}{bar:36}{r_bar}",
                )
            )
        del pool_inputs
        for i, (y, u, p) in enumerate(results):
            u_list[i, :, :] = u
            y_list[i, :, :] = y
        del results
    else:
        for i in tqdm(
            range(data_size), total=data_size, bar_format="{l_bar}{bar:36}{r_bar}"
        ):
            y, u, p = ssn(
                f_list[i, :, :],
                yd_list[i, :, :],
                alpha,
                ua_list[i, :, :],
                ub_list[i, :, :],
            )
            u_list[i, :, :] = u
            y_list[i, :, :] = y

    time_end = default_timer()
    print(f'Time taken: {time_end - time_start} seconds')
    print(f'Average time per sample: {(time_end - time_start) / data_size} seconds')

    write_filename = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'elliptic_u_y_test_res256_sz2048.npz',
    )
    with open(write_filename, 'wb') as file:
        np.savez(file, u=u_list, y=y_list)

    if debug:

        with open(write_filename, 'rb') as file:
            data = np.load(file)
            u_read, y_read = data['u'], data['y']
            data.close()
        print(f'Shape of u: {u_read.shape}')
        print(f'Shape of y: {y_read.shape}')

        x1, x2 = np.meshgrid(
            np.linspace(x_range[0], x_range[1], resol_space + 1),
            np.linspace(y_range[0], y_range[1], resol_space + 1),
        )

        for i in range(5):
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            surf = ax.plot_surface(  # type: ignore
                x1,
                x2,
                u_read[i, :, :],
                cmap=cm.coolwarm,  # type: ignore
                linewidth=0,
                antialiased=False,
            )
            plt.savefig(f'elliptic_u_{i}.pdf')
            plt.show()

            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            surf = ax.plot_surface(  # type: ignore
                x1,
                x2,
                y_read[i, :, :],
                cmap=cm.coolwarm,  # type: ignore
                linewidth=0,
                antialiased=False,
            )
            plt.savefig(f'elliptic_y_{i}.pdf')
            plt.show()
