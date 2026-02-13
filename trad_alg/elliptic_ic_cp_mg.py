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
    root_dir, 'data/elliptic_ic/elliptic_ic_f_yd_test_res256_sz64.npz'
)
read_filename_ua_ub = os.path.join(
    root_dir, 'data/elliptic_ic/elliptic_ic_ua_ub_test_res256_sz64.npz'
)
read_filename_u_y = os.path.join(
    root_dir, 'data/elliptic_ic/elliptic_ic_u_y_test_res256_sz64.npz'
)

data_size = 64
data_start = 0
max_iter_cp = 10000
tau = 2.0
sigma = 0.4

tol_rel = 4e-3

refinement_n = 8
resol_space = 2**refinement_n

alpha = 0.01


fem_setup = FEMSetupEllipticIC(refinement_n, alpha, use_lumped_mass=False)
mesh_init = fem_setup.mesh_init
mesh_now = fem_setup.mesh_now
fem = fem_setup.fem
vh = fem_setup.vh
free_node = fem_setup.free_node
node = fem_setup.node
elem = fem_setup.elem
mat_k = fem_setup.mat_k
mat_m = fem_setup.mat_m
num_free_node = fem_setup.num_free_node
num_node = fem_setup.num_node
norm_m = fem_setup.norm_m
norm_m_full = fem_setup.norm_m_full


mg_solver_k = MG(mesh_init, fem, refinement_n)
mg_solver_k.set_V(mat_k)
pre_k = lambda x: mg_solver_k.V_fun(x.reshape(-1, 1))
pre_k_oper = scipy.sparse.linalg.LinearOperator(
    (num_free_node, num_free_node),
    matvec=pre_k,  # type: ignore
    dtype=np.float64,  # type: ignore
)

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

#############################

time_start = default_timer()

for i in tqdm(
    range(data_start, data_start + data_size),
    total=data_size,
    bar_format='{l_bar}{bar:36}{r_bar}',
):
    fh = f_list[i]
    fh_free = fh[free_node, :]
    ydh = yd_list[i]
    ydh_free = ydh[free_node, :]
    uah = ua_list[i]
    ubh = ub_list[i]
    a = uah[free_node, :]
    b = ubh[free_node, :]
    ueh = u_exact_list[i]

    rel_err = np.inf
    uh_free = np.zeros((num_free_node, 1))
    yh_free = np.zeros((num_free_node, 1))
    ph_dual = np.zeros((num_free_node, 1))
    temp1 = np.zeros((num_free_node, 1))
    temp2 = np.zeros((num_free_node, 1))
    uh = np.zeros((num_node, 1))

    for i_cp in range(max_iter_cp):
        uh_old = uh_free
        yh_old = yh_free
        ph_old = ph_dual

        temp1 = scipy.sparse.linalg.cg(
            mat_k, mat_m @ ph_dual, x0=temp1, M=pre_k_oper, rtol=tol_rel
        )[0][:, None]
        uh_free = np.clip(
            (1.0 / (alpha * tau + 1.0)) * (uh_free - tau * temp1),
            a,
            b,
        )
        temp2 = scipy.sparse.linalg.cg(
            mat_k, mat_m @ (2.0 * uh_free - uh_old + fh_free), x0=temp2, M=pre_k_oper
        )[0][:, None]
        ph_dual = (1.0 / (1.0 + sigma)) * (ph_dual + sigma * temp2 - sigma * ydh_free)
        uh[free_node, :] = uh_free
        rel_err = norm_m_full(uh - ueh) / norm_m_full(ueh)
        iter_diff = np.max(
            [
                norm_m(uh_free - uh_old) / np.maximum(norm_m(uh_old), 1e-10),
                norm_m(yh_free - yh_old) / np.maximum(norm_m(yh_old), 1e-10),
                norm_m(ph_dual - ph_old) / np.maximum(norm_m(ph_old), 1e-10),
            ]
        )
        if rel_err < tol_rel or iter_diff < 1e-5:
            iter_list[i] = i_cp + 1
            break

    if iter_list[i] == 0:
        iter_list[i] = max_iter_cp

    err_list[i] = rel_err

time_end = default_timer()
print(f'Average time: {(time_end - time_start) / data_size:.4f} seconds')
print(f'Average relative error u: {np.mean(err_list):.4e}')
print(f'Average iteration: {np.mean(iter_list):.2f}')
