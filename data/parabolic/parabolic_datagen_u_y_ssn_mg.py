'''Import'''

import numpy as np
import scipy, scipy.sparse, scipy.sparse.linalg
import numpy.typing as npt
import functools
from timeit import default_timer
from tqdm import tqdm
import os, inspect

from utils.fem_mg.multigrid import MG, rearrange_by_nodes_batched
from trad_alg.fem_shared_setup import FEMSetupParabolic


''' Configs '''

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
root_dir = os.path.dirname(os.path.dirname(current_dir))
read_filename_f_yd = os.path.join(
    root_dir, 'data/parabolic/parabolic_f_yd_test_res128_sz64.npz'
)
write_filename_u_y = os.path.join(
    root_dir, 'data/parabolic/parabolic_u_y_test_res128_sz64.npz'
)

data_size = 4
max_iter_ssn = 10
tol_rel = 1e-6

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
mat_k: scipy.sparse.csc_matrix = fem_setup.mat_k
mat_kt = fem_setup.mat_kt
mat_kminvkt = fem_setup.mat_kminvkt
mat_l_inv = fem_setup.mat_l_inv
mat_lt_inv = fem_setup.mat_lt_inv
dot_m = fem_setup.dot_m
norm_m = fem_setup.norm_m
norm_m_full = fem_setup.norm_m_full


''' Global presolvers/preconditioners '''

mg_solver_list = [
    MG(mesh_init, fem, refinement_n) for _ in range(2 * (resol_time - 1))
]  # set as global for reusing factorization

mg_solver_k = MG(mesh_init, fem, refinement_n)
mg_solver_k.set_V(mat_m_lump_base / tau + mat_k_base)
pre_k = lambda x: mg_solver_k.V_fun(x[:, None]).flatten()
pre_k_oper = scipy.sparse.linalg.LinearOperator(
    shape=(num_free_node_base, num_free_node_base),
    matvec=pre_k,  # type: ignore
    dtype=np.float64,
)


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
    p = np.zeros(num_node)
    y = np.zeros(num_node)
    u = np.zeros(num_node)
    return y, u, p


def pre_cond(
    v: npt.NDArray, is_inactive_nodes: npt.NDArray, is_solver_reusable: npt.NDArray
) -> npt.NDArray:
    '''Preconditioner for the Schur complement system'''
    v1 = mg_solve(
        v[:num_free_node], is_inactive_nodes, is_solver_reusable=is_solver_reusable
    )
    v2 = mat_m_lump @ v1
    v3 = mgt_solve(v2, is_inactive_nodes, is_solver_reusable=is_solver_reusable)
    return v3


def mg_solve(
    v: npt.NDArray, is_inactive_nodes: npt.NDArray, is_solver_reusable: npt.NDArray
) -> npt.NDArray:
    '''Multigrid V-cycle solver for (K + M@Pi/sqrt(alpha))x = v'''
    sol = np.zeros_like(v)
    sol_curr = np.zeros(num_free_node_base)
    for i in range(resol_time - 1):
        idx_slice = slice(i * num_free_node_base, (i + 1) * num_free_node_base)
        matrix_pi = scipy.sparse.diags(
            is_inactive_nodes[idx_slice].astype(float), format='csc'
        )
        L1 = (
            mat_k_base
            + mat_m_lump_base / tau
            + mat_m_lump_base @ matrix_pi / np.sqrt(alpha)
        )
        if not is_solver_reusable[i]:
            mg_solver_list[i].set_V(L1)
        precond_S = lambda x: mg_solver_list[i].V_fun(x[:, None]).flatten()
        sol_curr = precond_S(
            v[idx_slice] if i == 0 else v[idx_slice] + mat_m_lump_base @ sol_curr / tau
        )
        sol[idx_slice] = sol_curr
    return sol


def mgt_solve(
    v: npt.NDArray, is_inactive_nodes: npt.NDArray, is_solver_reusable: npt.NDArray
) -> npt.NDArray:
    '''Multigrid V-cycle solver for (K + M@Pi/sqrt(alpha)).T x = v'''
    sol = np.zeros_like(v)
    sol_curr = np.zeros(num_free_node_base)
    for i in range(resol_time - 2, -1, -1):
        idx_slice = slice(i * num_free_node_base, (i + 1) * num_free_node_base)
        matrix_pi = scipy.sparse.diags(
            is_inactive_nodes[idx_slice].astype(float), format='csc'
        )
        L1 = (
            mat_k_base
            + mat_m_lump_base / tau
            + mat_m_lump_base @ matrix_pi / np.sqrt(alpha)
        )
        if not is_solver_reusable[i]:
            mg_solver_list[i + (resol_time - 1)].set_V(L1)
        pre_s = (
            lambda x: mg_solver_list[i + (resol_time - 1)].V_fun(x[:, None]).flatten()
        )
        sol_curr = pre_s(
            v[idx_slice]
            if i == resol_time - 2
            else v[idx_slice] + mat_m_lump_base @ sol_curr / tau
        )
        sol[idx_slice] = sol_curr
    return sol


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
    sol_prev: npt.NDArray | None = None,
    iin_prev: npt.NDArray | None = None,
    res: (
        tuple[npt.NDArray, npt.NDArray, npt.NDArray] | None
    ) = None,  # (res_primal, res_dual, obj_gap)
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

    iin = (p[free_node] > -beta - alpha * ub[free_node]) * (p[free_node] < -beta) + (
        p[free_node] > beta
    ) * (
        p[free_node] < beta - alpha * ua[free_node]
    )  # indicate is inactive node
    if iin_prev is not None:
        iin_reshaped = iin.reshape(resol_time - 1, num_free_node_base)
        iin_prev_reshaped = iin_prev.reshape(resol_time - 1, num_free_node_base)
        isu = np.all(iin_reshaped == iin_prev_reshaped, axis=1)
    else:
        isu = np.zeros(resol_time - 1, dtype=bool)
    matrix_var_block = (
        (-1.0 / alpha)
        * mat_m_lump
        @ scipy.sparse.diags(iin.astype(float), format='csc')
    )
    matrix_a = matrix_var_block - mat_kminvkt

    if res is not None:
        res_primal, res_dual, obj_gap = res
        rhs = np.hstack((res_dual, res_primal - mat_m_lump @ obj_gap))
    else:
        rhs = np.hstack(
            (
                mat_m_lump @ (y[free_node] - yd[free_node]) - mat_kt @ p[free_node],
                -mat_k @ y[free_node]
                + mat_m_lump
                @ (
                    psi_func(p[free_node], alpha, beta, ua[free_node], ub[free_node])
                    + f[free_node]
                ),
            )
        )
    temp1 = mat_l_inv @ rhs
    temp2_upper = mat_m_lump_inv @ temp1[:num_free_node]
    pc = functools.partial(pre_cond, is_inactive_nodes=iin, is_solver_reusable=isu)
    pc_oper = scipy.sparse.linalg.LinearOperator(
        shape=(num_free_node, num_free_node), matvec=pc, dtype=np.float64  # type: ignore
    )
    temp2_lower, _ = scipy.sparse.linalg.cg(
        -matrix_a, -temp1[num_free_node:], x0=sol_prev, M=pc_oper
    )
    y_p_step = mat_lt_inv @ np.hstack((temp2_upper, temp2_lower))
    y_step = y_p_step[:num_free_node]
    p_step = y_p_step[num_free_node:]
    psiin = p_step.copy()
    psiin[~iin] = 0.0
    u_step = (
        u[free_node]
        - psi_func(p[free_node], alpha, beta, ua[free_node], ub[free_node])
        - (1.0 / alpha) * mat_m_lump @ psiin
    )
    return y_step, u_step, p_step, temp2_lower, iin


def ssn(
    f: npt.NDArray,
    yd: npt.NDArray,
    alpha: float,
    beta: float,
    ua: npt.NDArray,
    ub: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    '''SSN applied on single instance of f, yd'''
    y, u, p = init_ssn()
    # sol_prev = None
    iin_prev = None
    res = None

    for _ in range(max_iter_ssn):
        y_step, u_step, p_step, _, iin_prev = get_ssn_step(
            y,
            u,
            p,
            f,
            yd,
            alpha,
            beta,
            ua,
            ub,
            sol_prev=None,
            iin_prev=iin_prev,
            res=res,
        )
        y[free_node] = y[free_node] - y_step
        u[free_node] = u[free_node] - u_step
        p[free_node] = p[free_node] - p_step

        res_dual = mat_m_lump @ (y[free_node] - yd[free_node]) - mat_kt @ p[free_node]
        res_primal = -mat_k @ y[free_node] + mat_m_lump @ (u[free_node] + f[free_node])
        obj_gap = u[free_node] - psi_func(
            p[free_node], alpha, beta, ua[free_node], ub[free_node]
        )
        res = (res_primal, res_dual, obj_gap)
        res_dual_rel = norm_m(res_dual) / np.max(
            (norm_m(mat_m_lump @ yd[free_node]), 1e-10)
        )
        res_primal_rel = norm_m(res_primal) / np.max(
            (norm_m(mat_m_lump @ f[free_node]), 1e-10)
        )
        obj_gap_rel = norm_m(obj_gap) / np.max(
            (norm_m(p[free_node] / alpha), norm_m(u[free_node]), 1e-10)
        )
        err_rel = np.max((res_dual_rel, res_primal_rel, obj_gap_rel))
        # print(err_rel)
        if err_rel < tol_rel:
            break

    y[-num_node_base:][free_node_base], _ = scipy.sparse.linalg.cg(
        mat_m_lump_base / tau + mat_k_base,
        mat_m_lump_base
        @ (
            y[(-2 * num_node_base) : (-num_node_base)][free_node_base] / tau
            + f[-num_node_base:][free_node_base]
        ),  # if not ua < 0 < ub, then adding ua or ub here is necessary
        M=pre_k_oper,
    )
    p[:num_node_base][free_node_base], _ = scipy.sparse.linalg.cg(
        mat_m_lump_base / tau + mat_k_base,
        mat_m_lump_base
        @ (
            p[num_node_base : (2 * num_node_base)][free_node_base] / tau
            - yd[:num_node_base][free_node_base]
        ),
        M=pre_k_oper,
    )
    u = psi_func(p, alpha, beta, ua, ub)

    return y, u, p


''' Main function '''

if __name__ == '__main__':

    with open(read_filename_f_yd, 'rb') as file:
        data = np.load(file)
        f_list, yd_list = rearrange_by_nodes_batched(
            data['f'], node_base
        ), rearrange_by_nodes_batched(data['yd'], node_base)
        data.close()
    ua = -6.0 * np.ones_like(f_list[0])
    ub = 6.0 * np.ones_like(f_list[0])

    u_list = np.zeros((data_size, num_node))
    y_list = np.zeros((data_size, num_node))

    time_start = default_timer()
    for i in tqdm(
        range(data_size), total=data_size, bar_format='{l_bar}{bar:36}{r_bar}'
    ):
        y, u, p = ssn(f_list[i], yd_list[i], alpha, beta, ua, ub)
        y_list[i] = y
        u_list[i] = u
    time_end = default_timer()

    with open(write_filename_u_y, 'wb') as file:
        np.savez(file, u=u_list, y=y_list)

    print(f'Average time: {(time_end - time_start) / data_size:.4f} seconds')
