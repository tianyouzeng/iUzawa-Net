'''Shared FEM and multigrid setup'''

import numpy as np
import scipy, scipy.sparse

import utils.fem_mg.mesh as mesh
from utils.fem_mg.finite_element import FiniteElementLP1
from utils.fem_mg.oper_vec import (
    Quad,
    stiffness_matrix_first_order,
    mass_matrix,
)
from utils.fem_mg.multigrid import MG


class FEMSetupElliptic:
    def __init__(self, refinement_N: int, alpha: float, use_lumped_mass: bool = False):
        '''FEM and multigrid setup for elliptic problems'''

        quad2d = Quad()
        self.mesh_init = mesh.Mesh()
        coarse_partition = mesh.CoarsePartitionDirichlet()
        self.mesh_init = mesh.initial_mesh(coarse_partition, self.mesh_init)
        self.fem = FiniteElementLP1('LP1')
        mg_solver = MG(
            self.mesh_init, self.fem, refinement_N
        )  # this multigrid solver instance is only used for constructing mesh, do not use it in the scripts
        self.vh = mg_solver.Vh
        self.mesh_now = self.vh.space_mesh
        self.node = self.vh.space_mesh.node
        self.elem = self.vh.space_mesh.elem
        self.free_node = mg_solver.freeDOF
        self.num_free_node = len(self.free_node)
        self.num_node = self.vh.space_mesh.node.shape[0]

        mat_k_full = stiffness_matrix_first_order(self.vh, self.vh, quad2d)
        mat_m_full = mass_matrix(self.vh, self.vh, quad2d)

        self.mat_k = mat_k_full[self.free_node, :][:, self.free_node]

        node_masses = np.array(mat_m_full.sum(axis=1))[:, 0]
        mat_w_full = scipy.sparse.diags(node_masses).tocsr()
        self.mat_m_lump = mat_w_full[self.free_node, :][:, self.free_node]
        self.mat_m = mat_m_full[self.free_node, :][:, self.free_node]

        self.mat_m_diag_vec = self.mat_m.diagonal().reshape(
            -1, 1
        )  # note that diag_M depends on whether lumped mass is used
        self.mat_m_lump_diag_vec = self.mat_m_lump.diagonal().reshape(-1, 1)
        self.mat_m_lump_inv_diag_vec = scipy.sparse.diags(
            1.0 / self.mat_m_lump_diag_vec[:, 0]
        )

        # SSN specific
        self.mat_a = scipy.sparse.bmat(
            [
                [
                    self.mat_m_lump,
                    scipy.sparse.csr_matrix((self.num_free_node, self.num_free_node)),
                ],
                [
                    scipy.sparse.csr_matrix((self.num_free_node, self.num_free_node)),
                    alpha * self.mat_m_lump,
                ],
            ]
        )
        self.mat_a_inv = lambda r: np.concatenate(
            (
                r[: self.num_free_node] / self.mat_m_lump_diag_vec[:, 0],
                r[self.num_free_node :] / (alpha * self.mat_m_lump_diag_vec[:, 0]),
            )
        )

        # Inner product and norm
        self.dot_m = lambda x, y: (x.T @ (self.mat_m @ y)).item()
        self.norm_m = lambda x: np.sqrt(self.dot_m(x, x))
        self.dot_m_full = lambda x, y: (x.T @ (mat_m_full @ y)).item()
        self.norm_m_full = lambda x: np.sqrt(self.dot_m_full(x, x))


class FEMSetupEllipticIC:
    def __init__(self, refinement_N: int, alpha: float, use_lumped_mass: bool = False):
        '''FEM and multigrid setup for elliptic problems with anisotropic operator'''

        quad2d = Quad()
        self.mesh_init = mesh.Mesh()
        coarse_partition = mesh.CoarsePartitionNeumann()
        self.mesh_init = mesh.initial_mesh(coarse_partition, self.mesh_init)
        self.fem = FiniteElementLP1('LP1')
        mg_solver = MG(
            self.mesh_init, self.fem, refinement_N
        )  # this multigrid solver instance is only used for constructing mesh, do not use it in the scripts
        self.vh = mg_solver.Vh
        self.mesh_now = self.vh.space_mesh
        self.node = self.vh.space_mesh.node
        self.elem = self.vh.space_mesh.elem
        self.free_node = mg_solver.freeDOF
        self.num_free_node = len(self.free_node)
        self.num_node = self.vh.space_mesh.node.shape[0]

        mat_k_full = stiffness_matrix_first_order(
            self.vh, self.vh, quad2d, coeff_mat=np.array([[1.0, 0.0], [0.0, 1e2]])
        )
        mat_m_full = mass_matrix(self.vh, self.vh, quad2d)
        mat_k_full = mat_k_full + mat_m_full  # A + I, where A is the elliptic operator

        self.mat_k = mat_k_full[self.free_node, :][:, self.free_node]

        node_masses = np.array(mat_m_full.sum(axis=1))[:, 0]
        mat_w_full = scipy.sparse.diags(node_masses).tocsr()
        self.mat_m_lump = mat_w_full[self.free_node, :][:, self.free_node]
        self.mat_m = mat_m_full[self.free_node, :][:, self.free_node]

        self.mat_m_diag_vec = self.mat_m.diagonal().reshape(
            -1, 1
        )  # note that diag_M depends on whether lumped mass is used
        self.mat_m_lump_diag_vec = self.mat_m_lump.diagonal().reshape(-1, 1)
        self.mat_m_lump_inv_diag_vec = scipy.sparse.diags(
            1.0 / self.mat_m_lump_diag_vec[:, 0]
        )

        # SSN specific
        self.mat_a = scipy.sparse.bmat(
            [
                [
                    self.mat_m,
                    scipy.sparse.csr_matrix((self.num_free_node, self.num_free_node)),
                ],
                [
                    scipy.sparse.csr_matrix((self.num_free_node, self.num_free_node)),
                    alpha * self.mat_m,
                ],
            ]
        )
        self.mat_a_inv = lambda r: np.concatenate(
            (
                r[: self.num_free_node] / self.mat_m_lump_diag_vec[:, 0],
                r[self.num_free_node :] / (alpha * self.mat_m_lump_diag_vec[:, 0]),
            )
        )

        # Inner product and norm
        self.dot_m = lambda x, y: (x.T @ (self.mat_m @ y)).item()
        self.norm_m = lambda x: np.sqrt(self.dot_m(x, x))
        self.dot_m_full = lambda x, y: (x.T @ (mat_m_full @ y)).item()
        self.norm_m_full = lambda x: np.sqrt(self.dot_m_full(x, x))


class FEMSetupParabolic:
    def __init__(self, refinement_N: int, resol_time: int, tau: float):
        '''FEM and multigrid setup for parabolic problems'''

        quad2d = Quad()
        self.mesh_init = mesh.Mesh()
        coarse_partition = mesh.CoarsePartitionDirichlet()
        self.mesh_init = mesh.initial_mesh(coarse_partition, self.mesh_init)
        self.fem = FiniteElementLP1('LP1')
        mg_solver = MG(
            self.mesh_init, self.fem, refinement_N
        )  # this multigrid solver instance is only used for constructing mesh, do not use it in the scripts
        self.vh = mg_solver.Vh
        self.mesh_now = self.vh.space_mesh
        self.node_base = self.vh.space_mesh.node
        self.elem_base = self.vh.space_mesh.elem
        self.num_node_base = len(self.node_base)
        self.num_node = self.num_node_base * (resol_time + 1)
        free_node_base_idx_list = mg_solver.freeDOF
        self.free_node_base = np.zeros(self.num_node_base, dtype=bool)
        self.free_node_base[free_node_base_idx_list] = True
        self.free_node = np.hstack(
            (
                np.zeros_like(self.free_node_base).astype(bool),
                np.tile(self.free_node_base, resol_time - 1),
                np.zeros_like(self.free_node_base).astype(bool),
            )
        )
        self.num_free_node_base = np.sum(self.free_node_base)
        self.num_free_node = np.sum(self.free_node)

        mat_k_base_full = stiffness_matrix_first_order(self.vh, self.vh, quad2d)
        mat_m_base_full = mass_matrix(self.vh, self.vh, quad2d)
        node_masses = np.array(mat_m_base_full.sum(axis=1))[:, 0]
        mat_w_base_full = scipy.sparse.diags(node_masses).tocsr()
        self.mat_k_base = mat_k_base_full[self.free_node_base, :][
            :, self.free_node_base
        ]
        mat_m_base = mat_w_base_full[self.free_node_base, :][:, self.free_node_base]

        # Mass lumping
        mat_m_rowsum = mat_m_base.sum(axis=1).A1
        self.mat_m_lump_base = scipy.sparse.diags(mat_m_rowsum)
        self.mat_m_lump_inv_base = scipy.sparse.diags(1.0 / mat_m_rowsum)

        # Block diagonal matrices for time-stepping
        self.mat_m_lump = scipy.sparse.block_diag(
            [self.mat_m_lump_base] * (resol_time - 1), format='csc'
        )
        self.mat_m_lump_inv = scipy.sparse.block_diag(
            [self.mat_m_lump_inv_base] * (resol_time - 1), format='csc'
        )
        mat_k_blockdiag = scipy.sparse.block_diag(
            [self.mat_k_base + self.mat_m_lump_base / tau] * (resol_time - 1),
            format='csc',
        )
        mat_k_subblockdiag = scipy.sparse.kron(
            scipy.sparse.diags(
                [np.ones(resol_time - 2)],
                offsets=[-1],  # type: ignore
                shape=(resol_time - 1, resol_time - 1),
                format='csc',
            ),
            -self.mat_m_lump_base / tau,
        )
        self.mat_k = mat_k_blockdiag + mat_k_subblockdiag
        self.mat_kt = self.mat_k.transpose()

        # Auxiliary matrices for Schur complement
        self.mat_l_inv = scipy.sparse.vstack(
            (
                scipy.sparse.hstack(
                    (
                        scipy.sparse.identity(self.num_free_node),
                        scipy.sparse.csc_matrix(
                            (self.num_free_node, self.num_free_node)
                        ),
                    )
                ),
                scipy.sparse.hstack(
                    (
                        self.mat_k @ self.mat_m_lump_inv,
                        scipy.sparse.identity(self.num_free_node),
                    )
                ),
            )
        )
        self.mat_lt_inv = self.mat_l_inv.transpose()
        self.mat_kminvkt = self.mat_k @ self.mat_m_lump_inv @ self.mat_kt

        # Inner product and norm
        mat_m = scipy.sparse.block_diag([mat_m_base] * (resol_time - 1), format='csc')
        self.dot_m = lambda x, y: (x.T @ (mat_m @ y)).squeeze() / (resol_time - 1)
        self.norm_m = lambda x: np.sqrt(self.dot_m(x, x))
        mat_m_full = scipy.sparse.block_diag(
            [mat_m_base_full] * (resol_time + 1), format='csc'
        )
        self.dot_m_full = lambda x, y: (x.T @ (mat_m_full @ y)).squeeze() / resol_time
        self.norm_m_full = lambda x: np.sqrt(self.dot_m_full(x, x))
