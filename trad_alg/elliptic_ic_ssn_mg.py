# author: Hangrui Yue
# modified by: Tianyou Zeng

import numpy as np
import scipy, scipy.sparse, scipy.sparse.linalg
from utils.fem_mg.multigrid import MG, rearrange_by_nodes_batched
from timeit import default_timer
from tqdm import tqdm
import os, inspect
from fem_shared_setup import FEMSetupEllipticIC


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
root_dir = os.path.dirname(current_dir)
read_filename_f_yd = os.path.join(
    root_dir, 'data/elliptic_ic/elliptic_ic_f_yd_test_res64_sz2048.npz'
)
read_filename_ua_ub = os.path.join(
    root_dir, 'data/elliptic_ic/elliptic_ic_ua_ub_test_res64_sz2048.npz'
)
read_filename_u_y = os.path.join(
    root_dir, 'data/elliptic_ic/elliptic_ic_u_y_test_res64_sz2048.npz'
)

data_size = 2048
max_iter_ssn = 10
tol_rel = 4e-3

refinement_n = 6
resol_space = 2**refinement_n

alpha = 0.01


fem_setup = FEMSetupEllipticIC(refinement_n, alpha, use_lumped_mass=True)
mesh_init = fem_setup.mesh_init
fem = fem_setup.fem
vh = fem_setup.vh
free_node = fem_setup.free_node
node = fem_setup.node
elem = fem_setup.elem
mat_k = fem_setup.mat_k
mat_m = fem_setup.mat_m_lump
num_free_node = fem_setup.num_free_node
num_node = fem_setup.num_node
mat_m_lump_diag_vec = fem_setup.mat_m_lump_diag_vec
mat_m_lump_inv_diag_vec = fem_setup.mat_m_lump_inv_diag_vec
mat_a = fem_setup.mat_a
mat_a_inv = fem_setup.mat_a_inv
norm_m = fem_setup.norm_m
norm_m_full = fem_setup.norm_m_full


c = 1
mg_solver_s = MG(mesh_init, fem, refinement_n)


with open(read_filename_f_yd, 'rb') as file:
    data = np.load(file)
    f_list, yd_list = (
        rearrange_by_nodes_batched(data['f'][..., None], node)[..., None],
        rearrange_by_nodes_batched(data['yd'][..., None], node)[..., None],
    )
    data.close()

with open(read_filename_ua_ub, 'rb') as file:
    data = np.load(file)
    ua_list, ub_list = (
        rearrange_by_nodes_batched(data['ua'][..., None], node)[..., None],
        rearrange_by_nodes_batched(data['ub'][..., None], node)[..., None],
    )
    data.close()

with open(read_filename_u_y, 'rb') as file:
    data = np.load(file)
    u_exact_list, y_exact_list = (
        rearrange_by_nodes_batched(data['u'][..., None], node)[..., None],
        rearrange_by_nodes_batched(data['y'][..., None], node)[..., None],
    )
    data.close()

u_list = np.zeros((data_size, resol_space + 1, resol_space + 1))
y_list = np.zeros((data_size, resol_space + 1, resol_space + 1))
err_list = np.zeros(data_size)
iter_list = np.zeros(data_size, dtype=int)


t1 = default_timer()

for i in tqdm(range(data_size), total=data_size, bar_format='{l_bar}{bar:36}{r_bar}'):
    fh = f_list[i]
    fh_free = fh[free_node, :]
    ydh = yd_list[i]
    ydh_free = ydh[free_node, :]
    uah = ua_list[i]
    ubh = ub_list[i]
    a = uah[free_node, 0]
    b = ubh[free_node, 0]
    ueh = u_exact_list[i]

    uh_free = np.zeros((num_free_node, 1))
    yh_free = np.zeros((num_free_node, 1))
    ph_dual = np.zeros((num_free_node, 1))
    muh_dual = np.zeros((num_free_node, 1))
    uh = np.zeros((num_node, 1))
    rel_err = np.inf

    for i_ssn in range(max_iter_ssn):
        uh_old = uh_free.copy()
        yh_old = yh_free.copy()
        ph_old = ph_dual.copy()
        muh_old = muh_dual.copy()

        c_a = muh_dual[:, 0] + c * (uh_free[:, 0] - a)
        c_b = muh_dual[:, 0] + c * (uh_free[:, 0] - b)
        is_active = (c_a < 0).astype(int) + (c_b > 0).astype(int)
        is_inactive = (
            is_active == 0
        )  # bool arrays are set to True by default, so no need to use np.ones
        where_active_minus = np.where(c_a < 0)[0]
        where_active_plus = np.where(c_b > 0)[0]
        where_active = np.where(is_active == 1)[0]
        num_active = len(where_active)

        mat_active = scipy.sparse.csr_matrix(
            (np.ones(num_active), (np.arange(num_active), where_active)),
            shape=(num_active, num_free_node),
        )
        result = mat_m_lump_inv_diag_vec @ is_active  # use sparse matrix multiplication
        ker_mm = result[
            result.nonzero()
        ].flatten()  # obtain non-zero elements and convert to dense format
        improd = mat_m_lump_diag_vec.copy()
        improd[~is_inactive] = 0
        mat_l = scipy.sparse.diags(np.array(improd)[:, 0]).tocsr()
        mat_l1 = np.sqrt(alpha) * mat_k + mat_l
        num_free_node = len(free_node)

        L_aux, L_Smoother_aux, invL_c = mg_solver_s.set_V(mat_l1)
        pre_s = lambda x: mg_solver_s.V_fun(
            x.reshape(-1, 1), L_aux, L_Smoother_aux, invL_c
        ).flatten()
        pre_s_op = scipy.sparse.linalg.LinearOperator(
            (num_free_node, num_free_node),
            matvec=pre_s,  # type: ignore
            dtype=np.float64,  # type: ignore
        )
        pre_s_oper = lambda x: scipy.sparse.linalg.gmres(mat_l1, x, M=pre_s_op)[0]

        def s_inv_func(r):
            if len(r) > num_free_node:
                r_1 = r[:num_free_node] + mat_active.T * (r[num_free_node:] / ker_mm)
                r_2 = (
                    r[num_free_node:] / ker_mm
                    + (
                        mat_active
                        * pre_s_oper(mat_m_lump_diag_vec[:, 0] * pre_s_oper(r_1))
                    )
                    / ker_mm
                )
                r = np.concatenate(
                    (pre_s_oper(mat_m_lump_diag_vec[:, 0] * pre_s_oper(r_1)), r_2)
                )
            else:
                r = pre_s_oper(mat_m_lump_diag_vec[:, 0] * pre_s_oper(r))
            return alpha * r

        def m_func(r):
            r_1 = mat_a_inv(r[: 2 * num_free_node])
            r_2 = r[2 * num_free_node :] - mat_b * r_1
            r_2 = -s_inv_func(r_2)
            return np.concatenate((r_1 - mat_a_inv(mat_b.T * r_2), r_2))

        muh_dual[is_inactive, 0] = 0
        mat_b = scipy.sparse.bmat(
            [
                [mat_k, -mat_m],
                [scipy.sparse.csr_matrix((num_active, num_free_node)), mat_active],
            ]
        )
        mat_j = scipy.sparse.bmat(
            [
                [mat_a, mat_b.transpose()],
                [
                    mat_b,
                    scipy.sparse.csr_matrix(
                        (num_free_node + num_active, num_free_node + num_active)
                    ),
                ],
            ]
        )
        temp = np.zeros((num_free_node, 1))
        temp[where_active_minus, 0] = a[where_active_minus]
        temp[where_active_plus, 0] = b[where_active_plus]
        f = np.concatenate(
            [
                (mat_m @ ydh_free)[:, 0],
                np.zeros((num_free_node, 1)).flatten(),
                # np.zeros((n_free, 1)).flatten(),
                (mat_m @ fh_free)[:, 0],
                temp[where_active, 0],
            ]
        )
        big_n = len(f)
        pre_m_oper = scipy.sparse.linalg.LinearOperator(
            (big_n, big_n),
            matvec=m_func,  # type: ignore
            dtype=np.float64,  # type: ignore
        )
        w_0 = np.concatenate(
            [yh_free[:, 0], uh_free[:, 0], ph_dual[:, 0], muh_dual[where_active, 0]]
        )
        relre_0 = np.linalg.norm(
            f - mat_j @ w_0
        )  # use @ for matrix-vector multiplication
        tol_inner = np.max([1e-10, 1e-10 * relre_0])  # type: ignore
        w, info = scipy.sparse.linalg.gmres(
            mat_j, f, x0=w_0, atol=tol_inner, maxiter=1000, M=pre_m_oper
        )
        yh_free = w[:num_free_node].reshape(-1, 1)
        uh_free = w[num_free_node : 2 * num_free_node].reshape(-1, 1)
        ph_dual = w[2 * num_free_node : 3 * num_free_node].reshape(-1, 1)
        if len(where_active) > 0:
            muh_dual[where_active, 0] = w[3 * num_free_node :]

        uh[free_node, :] = uh_free
        rel_err = norm_m_full(uh - ueh) / norm_m_full(ueh)
        iter_diff = np.max(
            [
                norm_m(uh_free - uh_old) / np.maximum(norm_m(uh_old), 1e-10),
            ]
        )
        if rel_err < tol_rel or iter_diff < 1e-5:
            iter_list[i] = i_ssn + 1
            break

    if iter_list[i] == 0:
        iter_list[i] = max_iter_ssn
        print(i)

    err_list[i] = rel_err

t2 = default_timer()
print(f'Average time: {(t2 - t1) / data_size:.4f} seconds')
print(f'Average relative error u: {np.mean(err_list):.4e}')
print(f'Average number of SSN iterations: {np.mean(iter_list):.2f}')
