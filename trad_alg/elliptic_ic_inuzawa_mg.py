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
max_iter_inuzawa = 1000
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
mat_m_lump = fem_setup.mat_m_lump
mat_m_lump_diag_vec = fem_setup.mat_m_lump_diag_vec
num_free_node = fem_setup.num_free_node
num_node = fem_setup.num_node
mat_m_diag_vec = fem_setup.mat_m_diag_vec
norm_m = fem_setup.norm_m
norm_m_full = fem_setup.norm_m_full

mat_s = 1 / np.sqrt(alpha) * mat_m + mat_k
mg_solver_s = MG(mesh_init, fem, refinement_n)
mg_solver_s.set_V(mat_s)
pre_s = lambda x: mg_solver_s.V_fun(x.reshape(-1, 1))
pre_s_oper = scipy.sparse.linalg.LinearOperator((num_free_node, num_free_node), matvec=pre_s, dtype=np.float64)  # type: ignore


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


time_start = default_timer()
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

    rel_err = np.inf
    uh_free = np.zeros((num_free_node, 1))
    yh_free = np.zeros((num_free_node, 1))
    ph_dual = np.zeros((num_free_node, 1))
    kr_1 = np.zeros((num_free_node, 1))
    kr_3 = np.zeros((num_free_node, 1))
    uh = np.zeros((num_node, 1))

    for i_inuzawa in range(max_iter_inuzawa):
        uh_old = uh_free
        yh_old = yh_free
        ph_old = ph_dual

        uh_free = np.clip(
            ((mat_m_lump - mat_m) @ uh_old - mat_m @ ph_old / alpha)
            / mat_m_lump_diag_vec,
            a.reshape(-1, 1),
            b.reshape(-1, 1),
        )
        by = mat_k.T @ ph_old + mat_m @ ydh_free
        yh_free = yh_free + (by - mat_m @ yh_old) / (2 * (mat_m_diag_vec))
        kr_1 = scipy.sparse.linalg.cg(
            mat_s,
            mat_m @ (uh_free + fh_free) - mat_k @ yh_free,
            M=pre_s_oper,
            x0=kr_1,
            rtol=tol_rel,
        )[0][:, None]
        kr_2 = mat_m @ kr_1
        kr_3 = scipy.sparse.linalg.cg(
            mat_s, kr_2, M=pre_s_oper, x0=kr_3, rtol=tol_rel
        )[0][:, None]
        ph_dual = ph_old + 1.0 * kr_3
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
            iter_list[i] = i_inuzawa + 1
            break

    if iter_list[i] == 0:
        iter_list[i] = max_iter_inuzawa
        print(i)
    err_list[i] = rel_err

time_end = default_timer()
print(f'Average time: {(time_end - time_start) / data_size:.4f} seconds')
print(f'Average relative error u: {np.mean(err_list):.4e}')
print(f'Average iterations: {np.mean(iter_list):.2f}')
