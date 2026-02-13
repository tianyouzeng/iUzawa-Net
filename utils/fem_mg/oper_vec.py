# In this part, we give the quadrature formula, operators and vectors in forming the discrete problem.

# Copyright @ Zhiyu Tan-Nov. 11, 2022.
# ==================================================================================
# Remark:
# a = np.zeros((4,3))
# a[:,[0,1]] := zeros(4,2)
# a[0:1, :] : = zeros(2,3)
# while a[:,0:1].T := a[:,0] for the data, by the data structure is still different.
# a[:,0:1].shape = (4,1) while a[:,0].shape = (4,)
# a[:, range(2)] = a[:,[0,1]] is the first two columns of a
# More details can be found:
# https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r

import numpy as np
from numpy.typing import NDArray


# a test polynomial
def f_polynomial(x):
    v = x[:, 1]
    return v


# the quadrature formula
class Quad:
    """
    This class defines the quadrature formula.
    """

    @staticmethod
    def quad2d_triangle27():
        """
        This function returns a quadrature formula with 27 quadrature points on the reference triangle.
        The output is a 27-by-3 array.
        order = 11
        """
        v = np.array([[0.323649481112758957701558415465115104e-1, 0.323649481112758957701558415465115104e-1,
                       0.136597310026778627489729700528187095e-1],
                      [0.323649481112758957701558415465115104e-1, 0.935270103777448236215263932535890490,
                       0.136597310026778627489729700528187095e-1],
                      [0.935270103777448236215263932535890490, 0.323649481112758957701558415465115104e-1,
                       0.136597310026778627489729700528187095e-1],
                      [0.119350912282581309020734749992698198, 0.119350912282581309020734749992698198,
                       0.361845405034180792047671104683104204e-1],
                      [0.119350912282581309020734749992698198, 0.761298175434837354202954884385690093,
                       0.361845405034180792047671104683104204e-1],
                      [0.761298175434837354202954884385690093, 0.119350912282581309020734749992698198,
                       0.361845405034180792047671104683104204e-1],
                      [0.534611048270758337785935054853325710, 0.534611048270758337785935054853325710,
                       0.927006328960676049232014772627508137e-3],
                      [0.534611048270758337785935054853325710, -0.692220965415166200607188784488243982e-1,
                       0.927006328960676049232014772627508137e-3],
                      [-0.692220965415166200607188784488243982e-1, 0.534611048270758337785935054853325710,
                       0.927006328960676049232014772627508137e-3],
                      [0.203309900431282469801530510267184582, 0.203309900431282469801530510267184582,
                       0.593229773807740709545832658022845862e-1],
                      [0.203309900431282469801530510267184582, 0.593380199137435004885787748207803816,
                       0.593229773807740709545832658022845862e-1],
                      [0.593380199137435004885787748207803816, 0.203309900431282469801530510267184582,
                       0.593229773807740709545832658022845862e-1],
                      [0.398969302965855199261113739339634776, 0.398969302965855199261113739339634776,
                       0.771495349148131198679934072970354464e-1],
                      [0.398969302965855199261113739339634776, 0.202061394068289545966621290062903427,
                       0.771495349148131198679934072970354464e-1],
                      [0.202061394068289545966621290062903427, 0.398969302965855199261113739339634776,
                       0.771495349148131198679934072970354464e-1],
                      [0.593201213428212748013379496114794165, 0.501781383104946618334274432982056169e-1,
                       0.523371119622040720242850397880829405e-1],
                      [0.501781383104946618334274432982056169e-1, 0.356620648261292583214299156679771841,
                       0.523371119622040720242850397880829405e-1],
                      [0.356620648261292583214299156679771841, 0.593201213428212748013379496114794165,
                       0.523371119622040720242850397880829405e-1],
                      [0.593201213428212748013379496114794165, 0.356620648261292583214299156679771841,
                       0.523371119622040720242850397880829405e-1],
                      [0.501781383104946618334274432982056169e-1, 0.593201213428212748013379496114794165,
                       0.523371119622040720242850397880829405e-1],
                      [0.356620648261292583214299156679771841, 0.501781383104946618334274432982056169e-1,
                       0.523371119622040720242850397880829405e-1],
                      [0.807489003159792106956160750996787101, 0.210220165361662963965372341590409633e-1,
                       0.207076596391406880792729339191282634e-1],
                      [0.210220165361662963965372341590409633e-1, 0.171488980304041555013938591400801670,
                       0.207076596391406880792729339191282634e-1],
                      [0.171488980304041555013938591400801670, 0.807489003159792106956160750996787101,
                       0.207076596391406880792729339191282634e-1],
                      [0.807489003159792106956160750996787101, 0.171488980304041555013938591400801670,
                       0.207076596391406880792729339191282634e-1],
                      [0.210220165361662963965372341590409633e-1, 0.807489003159792106956160750996787101,
                       0.207076596391406880792729339191282634e-1],
                      [0.171488980304041555013938591400801670, 0.210220165361662963965372341590409633e-1,
                       0.207076596391406880792729339191282634e-1]])
        return v


# from Example import ContinuousProblemLap
from scipy.sparse import csr_matrix, find, triu
# from Finite_element import FiniteElement, FiniteElementLP1
# from mesh import Mesh, CoarsePartition, initial_mesh
from .fe_space import FESpace


# generate the matrices
def stiffness_matrix_first_order(fem_space1: FESpace, fem_space2: FESpace, quad: Quad, coeff_mat: NDArray[np.float64]|None = None):
    """
    This subroutine is used to compute the second order stiffness matrix.
    (\nabla, \nabla)
    fem_space1: the trial space
    fem_space2: the test space
    quad: the quadrature formula
    """
    if coeff_mat is None:
        coeff_mat = np.eye(2)
    # the quadrature formula
    quad2d = quad.quad2d_triangle27()
    # the number of the points
    Ni = quad2d.shape[0]
    # the dimension of the finite element spaces
    dim1 = fem_space1.space_dim
    dim2 = fem_space2.space_dim
    # the degrees of freedom
    free_deg1 = fem_space1.dof_management()  # NT-by-l_dim1 array
    free_deg2 = fem_space2.dof_management()  # NT-by-l_dim2 array
    # the area of the elements: NT-by-1 array(2D)
    area = fem_space1.space_mesh.transform_information()[0]
    # the dimension of the local finite element spaces
    l_dim1 = fem_space1.space_fem.total_local_dofs
    l_dim2 = fem_space2.space_fem.total_local_dofs
    # the stiffness matrix
    A = csr_matrix((dim2, dim1), dtype=np.double)
    # assembling the matrix
    for i in range(Ni):
        # compute the reference point
        x_node = (1 - quad2d[i, 0] - quad2d[i, 1]) * np.array([0, 0]) + \
                 quad2d[i, 0] * np.array([1, 0]) + quad2d[i, 1] * np.array([0, 1])
        # compute the gradient of basis functions at the reference point
        gv1 = fem_space1.fem_basis_functions_gradient_ref([], x_node)  # l_dim1-by-Nt-by-2 array
        gv2 = fem_space2.fem_basis_functions_gradient_ref([], x_node)  # l_dim2-by-Nt-by-2 array
        # accumulate the matrix
        for l in range(l_dim1):
            for k in range(l_dim2):
                # Apply the coefficient matrix: (a \nabla u) \cdot \nabla v
                # Note: We compute (gv1 @ coeff_mat.T) to get (a \nabla u) because gv1 rows are gradients
                # gu_rotated = gv1[l, :, :] @ coeff_mat.T
                vv = quad2d[i, 2] * np.sum((gv1[l, :, :] @ coeff_mat.T) * (gv2[k, :, :]), axis=1) * area.T[0]  # NT-by-1 array
                A = A + csr_matrix((vv, (free_deg2[:, k] - 1, free_deg1[:, l] - 1)), shape=(dim2, dim1),
                                   dtype=np.double)

    return A


def mass_matrix(fem_space1: FESpace, fem_space2: FESpace, quad: Quad):
    """
        This subroutine is used to compute the mass matrix.
        (cdot, cdot)
        fem_space1: the trial space
        fem_space2: the test space
        quad: the quadrature formula
        """
    # the quadrature formula
    quad2d = quad.quad2d_triangle27()
    # the number of the points
    Ni = quad2d.shape[0]
    # the dimension of the finite element spaces
    dim1 = fem_space1.space_dim
    dim2 = fem_space2.space_dim
    # the degrees of freedom
    free_deg1 = fem_space1.dof_management()  # NT-by-l_dim1 array
    free_deg2 = fem_space2.dof_management()  # NT-by-l_dim2 array
    # the area of the elements: NT-by-1 array(2D)
    area = fem_space1.space_mesh.transform_information()[0]
    # the dimension of the local finite element spaces
    l_dim1 = fem_space1.space_fem.total_local_dofs
    l_dim2 = fem_space2.space_fem.total_local_dofs
    # the stiffness matrix
    M = csr_matrix((dim2, dim1), dtype=np.double)
    # assembling the matrix
    for i in range(Ni):
        # compute the reference point
        x_node = (1 - quad2d[i, 0] - quad2d[i, 1]) * np.array([0, 0]) + \
                 quad2d[i, 0] * np.array([1, 0]) + quad2d[i, 1] * np.array([0, 1])
        # compute the gradient of basis functions at the reference point
        v1 = fem_space1.fem_basis_functions_value_ref([], x_node)  # NT-by-l_dim1 array
        v2 = fem_space2.fem_basis_functions_value_ref([], x_node)  # NT-by-l_dim2 array
        # accumulate the matrix
        for l in range(l_dim1):
            for k in range(l_dim2):
                vv = quad2d[i, 2] * v1[:, l] * v2[:, k] * area.T[0]  # NT-by-1 array
                M = M + csr_matrix((vv, (free_deg2[:, k] - 1, free_deg1[:, l] - 1)), shape=(dim2, dim1),
                                   dtype=np.double)

    return M


# generate the vectors
def righthand_side(f_r, fem_space: FESpace, quad: Quad):
    """
        This subroutine is used to compute the righthand side related to a given function.
        f_r : the given function
        fem_space2: the test space
        quad: the quadrature formula
        """
    # the quadrature formula
    quad2d = quad.quad2d_triangle27()
    # the number of the points
    Ni = quad2d.shape[0]
    # the dimension of the finite element space
    dim = fem_space.space_dim
    # the degrees of freedom
    free_deg = fem_space.dof_management()  # NT-by-l_dim1 array
    # the area of the elements: NT-by-1 array(2D)
    area = fem_space.space_mesh.transform_information()[0]
    # the dimension of the local finite element spaces
    l_dim = fem_space.space_fem.total_local_dofs
    # the righthand side
    F = csr_matrix((dim, 1), dtype=np.double)
    # the mesh
    mesh = fem_space.space_mesh
    # the number of elements
    NT = mesh.elem.shape[0]
    # assembling the vector
    for i in range(Ni):
        # compute the reference point
        x_node = (1 - quad2d[i, 0] - quad2d[i, 1]) * np.array([0, 0]) + \
                 quad2d[i, 0] * np.array([1, 0]) + quad2d[i, 1] * np.array([0, 1])
        # compute the gradient of basis functions at the reference point
        v = fem_space.fem_basis_functions_value_ref([], x_node)  # NT-by-l_dim1 array
        # the points on the elements: NT-by-2
        x_elem = (1 - quad2d[i, 0] - quad2d[i, 1]) * mesh.node[mesh.elem[:, 0] - 1, :] + \
                 quad2d[i, 0] * mesh.node[mesh.elem[:, 1] - 1, :] + quad2d[i, 1] * mesh.node[mesh.elem[:, 2] - 1, :]
        # the values of the given function at the points
        Fv = f_r(x_elem)
        # accumulate the vector
        for k in range(l_dim):
            vv = quad2d[i, 2] * v[:, k] * Fv * area.T[0]  # NT-by-1 array
            F = F + csr_matrix((vv, (free_deg[:, k] - 1, np.zeros(NT, dtype=int))), shape=(dim, 1), dtype=np.double)

    return F


# test
# quadrature formula
# quad = quad2d_triangle27()
# I_f = np.sum(f_polynomial(quad[:,[0,1]])*quad[:,2], axis = 0)
# I_f

# the coarse partition
# coarse_partition = CoarsePartition()
# the mesh container
# mesh0 = Mesh()
# the initial mesh
# mesh0 = initial_mesh(coarse_partition, mesh0)
# the finite element
# fem = FiniteElementLP1('LP1')
# the finite element space
# Vh = FESpace(fem, mesh0)
# the quadrature formula
# quad = Quad()
# the first order stiffness matrix
# A = stiffness_matrix_first_order(Vh, Vh, quad)
# print('The first order stiffness matrix is \n')
# print(A.toarray())
# the mass matrix
# M = mass_matrix(Vh, Vh, quad)
# print('The mass matrix is \n')
# print(M.toarray())
# the righthand side
# print('The righthand side is \n')
# F = righthand_side(f_polynomial, Vh, quad)
# print(F.toarray())
