'''Import'''

import numpy as np
import scipy, scipy.sparse, scipy.sparse.linalg
import numpy.typing as npt
from timeit import default_timer
from tqdm import tqdm
import os, inspect

from utils.fem_mg.multigrid import MG, rearrange_by_nodes_batched
from fem_shared_setup import FEMSetupParabolic


''' Configs '''

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
root_dir = os.path.dirname(current_dir)
read_filename_f_yd = os.path.join(
    root_dir, 'data/parabolic/parabolic_f_yd_test_res128_sz64.npz'
)
read_filename_u_y = os.path.join(
    root_dir, 'data/parabolic/parabolic_u_y_test_res128_sz64.npz'
)

data_size = 64
max_iter_cp = 50
tol_rel = 3e-3
tau_cp = 350.0
sigma_cp = 1.0

refinement_n = 7
resol_space = 2**refinement_n
resol_time = 128


''' Problem parameters '''

x_range = (0.0, 1.0)
y_range = (0.0, 1.0)
h = 1 / resol_space
tau = 1 / resol_time

alpha = 0.01
beta = 0.01


''' FEM and multigrid setup '''

fem_setup = FEMSetupParabolic(refinement_n, resol_time, tau)
vh = fem_setup.vh
mesh_init = fem_setup.mesh_init
mesh_now = fem_setup.mesh_now
fem = fem_setup.fem
node_base = fem_setup.node_base
num_node_base = fem_setup.num_node_base
num_node = fem_setup.num_node
free_node_base = fem_setup.free_node_base
free_node = fem_setup.free_node
num_free_node_base = fem_setup.num_free_node_base
num_free_node = fem_setup.num_free_node
mat_k_base = fem_setup.mat_k_base
mat_m_lump_base = fem_setup.mat_m_lump_base
mat_m_lump_inv_base = fem_setup.mat_m_lump_inv_base
mat_m_lump = fem_setup.mat_m_lump
mat_m_lump_inv = fem_setup.mat_m_lump_inv
mat_k = fem_setup.mat_k
mat_kt = fem_setup.mat_kt
mat_kminvkt = fem_setup.mat_kminvkt
mat_l_inv = fem_setup.mat_l_inv
mat_lt_inv = fem_setup.mat_lt_inv
dot_m = fem_setup.dot_m
norm_m = fem_setup.norm_m
norm_m_full = fem_setup.norm_m_full


''' Global presolvers/preconditioners '''

mg_solver_k = MG(fem_setup.mesh_init, fem_setup.fem, refinement_n)
mg_solver_k.set_V(mat_m_lump_base / tau + mat_k_base)
pre_k = lambda x: mg_solver_k.W_fun(x[:, None]).flatten()
pre_k_oper = scipy.sparse.linalg.LinearOperator(
    shape=(num_free_node_base, num_free_node_base),
    matvec=pre_k,  # type: ignore
    dtype=np.float64,
)


''' CP Iterations '''


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


def mg_solve(v: npt.NDArray) -> npt.NDArray:
    '''Multigrid V-cycle solver for Kx = v'''
    # v = v.astype(np.float64)
    sol = np.zeros_like(v)
    sol_curr = np.zeros(num_free_node_base)
    for i in range(resol_time - 1):
        idx_slice = slice(i * num_free_node_base, (i + 1) * num_free_node_base)
        sol_curr = pre_k(
            v[idx_slice] if i == 0 else v[idx_slice] + mat_m_lump_base @ sol_curr / tau
        )
        sol[idx_slice] = sol_curr
    return sol


def mgt_solve(v: npt.NDArray) -> npt.NDArray:
    sol = np.zeros_like(v)
    sol_curr = np.zeros(num_free_node_base)
    for i in range(resol_time - 2, -1, -1):
        idx_slice = slice(i * num_free_node_base, (i + 1) * num_free_node_base)
        sol_curr = pre_k(
            v[idx_slice]
            if (i == resol_time - 2)
            else v[idx_slice] + mat_m_lump_base @ sol_curr / tau
        )
        sol[idx_slice] = sol_curr
    return sol


def cp(
    f: npt.NDArray,
    yd: npt.NDArray,
    alpha: float,
    beta: float,
    ua: npt.NDArray,
    ub: npt.NDArray,
    u_exact: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, int, float]:
    u = np.zeros(num_node)
    y = np.zeros(num_node)
    p = np.zeros(num_node)
    iter_count = 0
    rel_err = np.inf

    for i in range(max_iter_cp):
        u_prev = u.copy()
        p_prev = p.copy()
        s_star_p_free = mgt_solve(mat_m_lump @ p[free_node])
        t = u[free_node] - tau_cp * s_star_p_free
        u[free_node] = np.clip(
            (1.0 / (alpha * tau_cp + 1.0))
            * (t + np.clip(-t, -tau_cp * beta, tau_cp * beta)),
            ua[free_node],
            ub[free_node],
        )
        p[free_node] = (1.0 / (1.0 + sigma_cp)) * (
            p[free_node]
            + sigma_cp
            * mg_solve(
                mat_m_lump @ (2.0 * u[free_node] - u_prev[free_node] + f[free_node])
            )
            - sigma_cp * yd[free_node]
        )

        rel_err = (
            norm_m(u[free_node] - u_exact[free_node]) / norm_m(u_exact[free_node])
            if u_exact is not None
            else np.inf
        )
        iter_diff = np.max(
            [
                norm_m(u[free_node] - u_prev[free_node])
                / np.maximum(norm_m(u_prev[free_node]), 1e-10),
                norm_m(p[free_node] - p_prev[free_node])
                / np.maximum(norm_m(p_prev[free_node]), 1e-10),
            ]
        )
        if rel_err < tol_rel or iter_diff < 1e-4:
            iter_count = i + 1
            break
    y[free_node] = mg_solve(mat_m_lump @ (u[free_node] + f[free_node]))

    # solution at t=0 and t=T
    y[-num_node_base:][free_node_base] = pre_k(
        mat_m_lump_base
        @ (
            y[(-2 * num_node_base) : (-num_node_base)][free_node_base] / tau
            + f[-num_node_base:][free_node_base]
        )  # if not ua < 0 < ub, then adding ua or ub here is necessary
    )
    p[:num_node_base][free_node_base] = (
        y[:num_node_base][free_node_base] - yd[:num_node_base][free_node_base]
    )  # in fact, -yd if homogeneous initial cond
    s_star_p_free = mgt_solve(mat_m_lump @ p[free_node])
    s_star_p_head = pre_k(
        mat_m_lump_base
        @ (
            s_star_p_free[num_free_node_base : (2 * num_free_node_base)] / tau
            - yd[:num_node_base][free_node_base]
        )
    )
    u[:num_node_base][free_node_base] = psi_func(
        s_star_p_head, alpha, beta, ua[:num_free_node_base], ub[:num_free_node_base]
    )

    if iter_count == 0:
        iter_count = max_iter_cp

    return y, u, p, iter_count, rel_err


''' Main function '''

if __name__ == '__main__':

    with open(read_filename_f_yd, 'rb') as file:
        data = np.load(file)
        f_list, yd_list = rearrange_by_nodes_batched(
            data['f'], node_base
        ), rearrange_by_nodes_batched(data['yd'], node_base)
        data.close()
    with open(read_filename_u_y, 'rb') as file:
        data = np.load(file)
        u_exact_list, y_exact_list = rearrange_by_nodes_batched(
            data['u'], node_base
        ), rearrange_by_nodes_batched(data['y'], node_base)
        data.close()
    ua = -6.0 * np.ones_like(f_list[0])
    ub = 6.0 * np.ones_like(f_list[0])

    u_list = np.zeros((data_size, num_node))
    y_list = np.zeros((data_size, num_node))
    err_list = np.zeros(data_size)
    iter_list = np.zeros(data_size, dtype=int)

    time_start = default_timer()

    for i in tqdm(
        range(data_size), total=data_size, bar_format='{l_bar}{bar:36}{r_bar}'
    ):
        y, u, p, iter_count, rel_err = cp(
            f_list[i], yd_list[i], alpha, beta, ua, ub, u_exact_list[i]
        )
        y_list[i] = y
        u_list[i] = u
        err_list[i] = norm_m_full(u - u_exact_list[i]) / norm_m_full(u_exact_list[i])
        iter_list[i] = iter_count

    time_end = default_timer()

    print(f'Average time: {(time_end - time_start) / data_size:.4f} seconds')
    print(f'Average relative error u: {np.mean(err_list):.4e}')
    print(f'Average iterations: {np.mean(iter_list):.2f}')
