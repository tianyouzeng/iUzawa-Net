'''
Generate training data for optimal control of elliptic PDEs
Optimal control and state variables
Solve by semismooth Newton method with FEM discretization
'''

''' Import '''

import numpy as np
import scipy.sparse as sp
import numpy.typing as npt
from pypardiso import PyPardisoSolver

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
beta = 0.01


''' Configs '''

data_range = (0, 2048)
data_size = data_range[1] - data_range[0]
max_iter_ssn = 8
use_multiprocessing = False
process_count = 1
debug = False

read_filename_f_yd = os.path.join(
    root_dir, 'data/parabolic/parabolic_f_yd_test_res32_sz2048.npz'
)
write_filename = os.path.join(
    root_dir, 'data/parabolic/parabolic_u_y_test_res32_sz2048.npz'
)


''' Solver '''

pypardiso_solver = PyPardisoSolver(mtype=2)


def spsolve(
    A, b, factorize=True, squeeze=True, solver=pypardiso_solver, *args, **kwargs
):
    if sp.issparse(A) and A.format == "csc":
        A = A.tocsr()  # fixes issue with brightway2 technosphere matrix

    solver._check_A(A)
    if factorize and not solver._is_already_factorized(A):
        solver.factorize(A)

    x = solver.solve(A, b)

    if squeeze:
        return (
            x.squeeze()
        )  # scipy spsolve always returns vectors with shape (n,) indstead of (n,1)
    else:
        return x


''' Space geometry '''

x_range = (0.0, 1.0)
y_range = (0.0, 1.0)
resol_space = 32
resol_time = 32
h = 1 / resol_space
tau = 1 / resol_time
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
free_node_base = ~is_bd_node
free_node_base_idx_list = free_node_base * np.cumsum(free_node_base) - 1
free_node = np.hstack(
    (
        np.zeros_like(free_node_base).astype(bool),
        np.tile(free_node_base, resol_time - 1),
        np.zeros_like(free_node_base).astype(bool),
    )
)
num_free_node_base = np.sum(free_node_base)
num_free_node = np.sum(free_node)


''' FEM matrices '''

# For most of the cases, only store the free_node submatrix is necessary for computation

# Stiff matrix
matrix_k = sp.csc_matrix((num_free_node_base, num_free_node_base))
for i in range(3):
    for j in range(3):
        matrix_k_ij = area * (der_phi[..., i] * der_phi[..., j]).sum(axis=-1)
        coords_i_temp = free_node_base_idx_list[elem[:, i]]
        coords_j_temp = free_node_base_idx_list[elem[:, j]]
        ij_is_free_node = (coords_i_temp != -1) & (coords_j_temp != -1)
        coords_i = coords_i_temp[ij_is_free_node]
        coords_j = coords_j_temp[ij_is_free_node]
        matrix_k += sp.csc_matrix(
            (matrix_k_ij[ij_is_free_node], (coords_i, coords_j)),
            shape=(num_free_node_base, num_free_node_base),
        )

# Mass matrix for the right hand side
matrix_m_full = sp.csc_matrix((num_free_node_base, num_free_node_base))
for i in range(3):
    for j in range(3):
        matrix_m_ij = (1.0 + (i == j)) / 12.0 * area
        coords_i_temp = free_node_base_idx_list[elem[:, i]]
        coords_j_temp = free_node_base_idx_list[elem[:, j]]
        ij_is_free_node = (coords_i_temp != -1) & (coords_j_temp != -1)
        coords_i = coords_i_temp[ij_is_free_node]
        coords_j = coords_j_temp[ij_is_free_node]
        matrix_m_full += sp.csc_matrix(
            (matrix_m_ij[ij_is_free_node], (coords_i, coords_j)),
            shape=(num_free_node_base, num_free_node_base),
        )

# Mass lumping
matrix_m_rowsum = matrix_m_full.sum(axis=1).A1
matrix_m = sp.diags(matrix_m_rowsum)
matrix_m_inv = sp.diags(1.0 / matrix_m_rowsum)

matrix_m_block = sp.block_diag([matrix_m] * (resol_time - 1), format='csc')
matrix_m_block_inv = sp.block_diag([matrix_m_inv] * (resol_time - 1), format='csc')
matrix_k_block_diag = sp.block_diag(
    [matrix_k + matrix_m / tau] * (resol_time - 1), format='csc'
)
matrix_k_block_subdiag = sp.kron(
    sp.diags([np.ones(resol_time - 2)], offsets=[-1], shape=(resol_time - 1, resol_time - 1), format='csc'),  # type: ignore
    -matrix_m / tau,
)
matrix_k_block = matrix_k_block_diag + matrix_k_block_subdiag
matrix_kt_block = matrix_k_block.transpose()

matrix_l_inv = sp.vstack(
    (
        sp.hstack(
            (sp.identity(num_free_node), sp.csc_matrix((num_free_node, num_free_node)))
        ),
        sp.hstack((matrix_k_block @ matrix_m_block_inv, sp.identity(num_free_node))),
    )
)
matrix_lt_inv = matrix_l_inv.transpose()
matrix_kminvkt_block = matrix_k_block @ matrix_m_block_inv @ matrix_kt_block


''' SSN Iterations '''


def psi_func(
    t: npt.NDArray,
    alpha: float,
    beta: float,
    ua: float | npt.NDArray,
    ub: float | npt.NDArray,
) -> npt.NDArray:
    '''Scalar valued function in the superposition operator'''
    psi = np.clip(-1.0 / alpha * (t + np.clip(-t, -beta, beta)), ua, ub)
    return psi


def init_ssn() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    p = np.zeros(num_node * (resol_time + 1))
    y = np.zeros(num_node * (resol_time + 1))
    u = np.zeros(num_node * (resol_time + 1))
    # u[free_node] = matrix_k * y[free_node]
    return y, u, p


def get_ssn_step(
    y: npt.NDArray,
    u: npt.NDArray,
    p: npt.NDArray,
    f: npt.NDArray,
    yd: npt.NDArray,
    alpha: float,
    beta: float,
    ua: npt.NDArray,
    ub: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:

    is_inactive_nodes = (p[free_node] > -beta - alpha * ub[free_node]) * (
        p[free_node] < -beta
    ) + (p[free_node] > beta) * (p[free_node] < beta - alpha * ua[free_node])
    matrix_var_block = (
        (-1.0 / alpha)
        * matrix_m_block
        @ sp.diags(is_inactive_nodes.astype(float), format='csc')
    )
    matrix_a = matrix_var_block - matrix_k_block @ matrix_m_block_inv @ matrix_kt_block

    # matrix_a = matrix_a.tocsc()
    rhs = np.hstack(
        (
            matrix_m_block @ (y[free_node] - yd[free_node])
            - matrix_kt_block @ p[free_node],
            -matrix_k_block @ y[free_node]
            + matrix_m_block
            @ (
                psi_func(p[free_node], alpha, beta, ua[free_node], ub[free_node])
                + f[free_node]
            ),
        )
    )
    temp1 = matrix_l_inv @ rhs
    temp2_upper = matrix_m_block_inv @ temp1[:num_free_node]
    temp2_lower = spsolve(sp.triu(-matrix_a, format='csr'), -temp1[num_free_node:])
    y_p_step = matrix_lt_inv @ np.hstack((temp2_upper, temp2_lower))
    y_step = y_p_step[:num_free_node]
    p_step = y_p_step[num_free_node:]
    return y_step, p_step


def ssn(
    f: npt.NDArray,
    yd: npt.NDArray,
    alpha: float,
    beta: float,
    ua: npt.NDArray,
    ub: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    '''SSN applied on single instance of f, yd'''
    f = f.transpose(2, 0, 1).reshape(-1)
    yd = yd.transpose(2, 0, 1).reshape(-1)
    ua = ua.transpose(2, 0, 1).reshape(-1)
    ub = ub.transpose(2, 0, 1).reshape(-1)
    y, u, p = init_ssn()

    for _ in range(max_iter_ssn):
        y_step, p_step = get_ssn_step(y, u, p, f, yd, alpha, beta, ua, ub)
        y[free_node] = y[free_node] - y_step
        # u[free_node] = u[free_node_p] - u_step
        p[free_node] = p[free_node] - p_step

    y[-num_node:][free_node_base] = spsolve(
        sp.triu(matrix_m / tau + matrix_k, format='csr'),
        matrix_m
        @ (
            y[(-2 * num_node) : (-num_node)][free_node_base] / tau
            + f[-num_node:][free_node_base]
        ),
    )
    p[:num_node][free_node_base] = spsolve(
        sp.triu(matrix_m / tau + matrix_k, format='csr'),
        matrix_m
        @ (
            p[num_node : (2 * num_node)][free_node_base] / tau
            - yd[:num_node][free_node_base]
        ),
    )
    u = psi_func(p, alpha, beta, ua, ub)

    u = u.reshape((resol_time + 1, resol_space + 1, resol_space + 1)).transpose(1, 2, 0)
    y = y.reshape((resol_time + 1, resol_space + 1, resol_space + 1)).transpose(1, 2, 0)
    p = p.reshape((resol_time + 1, resol_space + 1, resol_space + 1)).transpose(1, 2, 0)
    return y, u, p


def ssn_star(
    f_yd_tuple: tuple[npt.NDArray, npt.NDArray],
    alpha: float,
    beta: float,
    ua: npt.NDArray,
    ub: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    return ssn(f_yd_tuple[0], f_yd_tuple[1], alpha, beta, ua, ub)


''' Generate optimal control and state variables '''

if __name__ == '__main__':

    with open(read_filename_f_yd, 'rb') as file:
        data = np.load(file)
        f_list, yd_list = np.array(data['f'][slice(*data_range)]), np.array(
            data['yd'][slice(*data_range)]
        )  # deep copy
        data.close()
    assert f_list.shape[0] == yd_list.shape[0]
    if data_size == -1:
        data_size: int = f_list.shape[0]
    del data

    ua = -6.0 * np.ones((resol_space + 1, resol_space + 1, resol_time + 1))
    ub = 6.0 * np.ones((resol_space + 1, resol_space + 1, resol_time + 1))

    u_list = np.zeros((data_size, resol_space + 1, resol_space + 1, resol_time + 1))
    y_list = np.zeros((data_size, resol_space + 1, resol_space + 1, resol_time + 1))
    pool_map_func = functools.partial(ssn_star, alpha=alpha, beta=beta, ua=ua, ub=ub)

    time_start = default_timer()

    if use_multiprocessing:
        pool_inputs = [(f_list[i], yd_list[i]) for i in range(data_size)]
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
            u_list[i] = u
            y_list[i] = y
        del results
    else:
        for i in tqdm(
            range(data_size), total=data_size, bar_format="{l_bar}{bar:36}{r_bar}"
        ):
            y, u, p = ssn(f_list[i], yd_list[i], alpha, beta, ua, ub)
            u_list[i] = u
            y_list[i] = y

    time_end = default_timer()
    print(f'Time taken: {time_end - time_start} seconds')
    print(f'Average time per sample: {(time_end - time_start) / data_size} seconds')

    pypardiso_solver.free_memory(everything=True)

    with open(write_filename, 'wb') as file:
        np.savez(file, u=u_list, y=y_list)
