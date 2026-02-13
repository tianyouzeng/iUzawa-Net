# In this part, we define the finite element space

# Copyright @ Zhiyu Tan-Nov. 11, 2022.
# ==================================================================================
import numpy as np
from scipy.sparse import csr_matrix, find, triu
from .finite_element import FiniteElement, FiniteElementLP1
from .mesh import Mesh, CoarsePartitionDirichlet, initial_mesh


class FESpace:
    """
    This class defines the finite element space based on the given finite element and mesh.
    """

    def __init__(self, fem: FiniteElement, mesh: Mesh):
        """
        FEM is the finite element.
        mesh is the mesh.
        """
        # the finite element
        self.space_fem = fem
        # the mesh
        self.space_mesh = mesh
        # the dimension of the finite element
        dim = fem.dof[0] * mesh.node.shape[0] + fem.dof[1] * mesh.nd4ed.shape[0] + fem.dof[2] * mesh.elem.shape[0]
        self.space_dim = dim

    def dof_management(self):
        """
        This method gives the information of the degrees of freedom of the finite element space.
        """
        # ---------------------------------------------------------------------------
        # the finite element
        fem = self.space_fem
        # the mesh
        mesh = self.space_mesh
        # the number of nodes
        Nn = mesh.node.shape[0]
        # the number of elements
        NT = mesh.elem.shape[0]
        # the number of edges
        Ne = mesh.nd4ed.shape[0]
        # ---------------------------------------------------------------------------
        # the following part works in python 3.9 and numpy 1.23.4 but not for numpy 1.19.2
        free_base = np.hstack((np.kron(Nn * np.arange(fem.dof[0]), np.ones((1, 3))),
                               np.kron(Nn * fem.dof[0] + Ne * np.arange(fem.dof[1]), np.ones((1, 3))),
                               np.kron(Nn * fem.dof[0] + Ne * fem.dof[1] + NT * np.arange(fem.dof[2]),
                                       np.ones((1, 1)))))
        free_deg = np.ones((NT, 1)) @ free_base + np.hstack((np.kron(np.ones((1, fem.dof[0])), mesh.elem),
                                                             np.kron(np.ones((1, fem.dof[1])), mesh.elem),
                                                             np.kron(np.ones((1, fem.dof[2])),
                                                                     np.array([np.arange(NT) + 1]).T)))
        return free_deg.astype(int)

    def fem_basis_functions_value_ref(self, elem_index, x):
        """
        Remark: This part is highly finite element relative!!!
        This method is used to compute the value of basis functions at the reference point x.
        elem_index can be '[]' or an index array(1D) used to specific the element.
        The output is a Nt-by-3 array.
        """
        # ---------------------------------------------------------------------------
        # the finite element
        fem = self.space_fem
        # the mesh
        mesh = self.space_mesh
        # find the elements
        if len(elem_index) == 0:
            elem = mesh.elem
        else:
            elem = mesh.elem[elem_index, :]

        if fem.name == 'LP1':
            # the number of the elements
            Nt = elem.shape[0]
            # get the value of the basis functions on the reference element at x
            b_value = FiniteElementLP1.basis_functions_value(x)
            # compute the value of basis functions
            v = np.zeros((Nt, 3))
            v = np.ones((Nt, 1)) @ b_value
        else:
            v = np.array([])

        return v

    def fem_basis_functions_gradient_ref(self, elem_index, x):
        """
        Remark: This part is highly finite element relative!!!
        This method is used to compute the gradient of basis functions at the reference point x.
        elem_index can be '[]' or an index array(1D) used to specific the element.
        The output is a 3-by-Nt-by-2 array.
        """
        # ---------------------------------------------------------------------------
        # the finite element
        fem = self.space_fem
        # the mesh
        mesh = self.space_mesh
        # compute the transfer matrices
        inv_transf_all = mesh.transform_information()[1]
        # find the elements
        if len(elem_index) == 0:
            elem = mesh.elem
            inv_transf = inv_transf_all
        else:
            elem = mesh.elem[elem_index, :]
            inv_transf = inv_transf_all[:, elem_index, :]

        # check the type of the finite element
        if fem.name == 'LP1':
            # the number of the elements
            Nt = elem.shape[0]
            # get the value of the basis functions on the reference element at x
            b_gradient = FiniteElementLP1.basis_functions_gradient(x)
            # compute the gradient of basis functions
            g_v = np.zeros((3, Nt, 2))
            g_v[0, :, 0] = b_gradient[0, 0] * inv_transf[0, :, 0] + b_gradient[1, 0] * inv_transf[0, :, 1]
            g_v[0, :, 1] = b_gradient[0, 0] * inv_transf[1, :, 0] + b_gradient[1, 0] * inv_transf[1, :, 1]  # \nabla\varphi_1
            g_v[1, :, 0] = b_gradient[0, 1] * inv_transf[0, :, 0] + b_gradient[1, 1] * inv_transf[0, :, 1]
            g_v[1, :, 1] = b_gradient[0, 1] * inv_transf[1, :, 0] + b_gradient[1, 1] * inv_transf[1, :, 1]  # \nabla\varphi_2
            g_v[2, :, 0] = b_gradient[0, 2] * inv_transf[0, :, 0] + b_gradient[1, 2] * inv_transf[0, :, 1]
            g_v[2, :, 1] = b_gradient[0, 2] * inv_transf[1, :, 0] + b_gradient[1, 2] * inv_transf[1, :, 1]  # \nabla\varphi_3
        else:
            g_v = np.array([])

        return g_v

    def fem_basis_functions_hessian_ref(self, elem_index, x):
        """
        Remark: This part is highly finite element relative!!!
        This method is used to compute the Hessian of basis functions at the reference point x.
        elem_index can be '[]' or an index array(1D) used to specific the element.
        The output is a 3-by-Nt-by-4 array.
        """
        # ---------------------------------------------------------------------------
        # the finite element
        fem = self.space_fem
        # the mesh
        mesh = self.space_mesh
        # compute the transfer matrices
        inv_transf_all = mesh.transform_information()[1]
        # find the elements
        if len(elem_index) == 0:
            elem = mesh.elem
            inv_transf = inv_transf_all
        else:
            elem = mesh.elem[elem_index, :]
            inv_transf = inv_transf_all[:, elem_index, :]

        # check the type of the finite element
        if fem.name == 'LP1':
            # the number of the elements
            Nt = elem.shape[0]
            # compute the Hessian of basis functions
            H_v = np.zeros((3, Nt, 4))
            '''
            # a general approach which can be used to any finite element
            # get the value of the basis functions on the reference element at x
            b_hessian = FEM.basis_functions_hessian(x)
            # some auxiliary arrays
            BHB = np.zeros((3, Nt, 4))
            HB = np.zeros((3, Nt, 4))
            # part one: BHB
            BHB[0,:,:] = np.kron(np.ones((Nt,1)),b_hessian[:,0]) # the first base function
            BHB[1,:,:] = np.kron(np.ones((Nt,1)),b_hessian[:,1]) # the second base function
            BHB[2,:,:] = np.kron(np.ones((Nt,1)),b_hessian[:,2]) # the third base function
            for i in range(3):
                # Hessian matrix on the elements (part two: HB = B^{-T}(BHB))
                HB[i,:,0] = inv_transf[0,:,0]*BHB[i,:,0] + inv_transf[0,:,1]*BHB[i,:,1]  # the partial_{11}
                HB[i,:,1] = inv_transf[1,:,0]*BHB[i,:,0] + inv_transf[1,:,1]*BHB[i,:,1]  # the partial_{21}
                HB[i,:,2] = inv_transf[0,:,0]*BHB[i,:,2] + inv_transf[0,:,1]*BHB[i,:,3]  # the partial_{12}
                HB[i,:,3] = inv_transf[1,:,0]*BHB[i,:,2] + inv_transf[1,:,1]*BHB[i,:,3]  # the partial_{12}
                # Hessian matrix on the elements (part three: H = B^{-T}(BHB)B^{-1})
                H_v[i,:,0] = HB[i,:,0]*inv_transf[0,:,0] + HB[i,:,2]*inv_transf[0,:,1]  # the partial_{11}
                H_v[i,:,1] = HB[i,:,1]*inv_transf[0,:,0] + HB[i,:,3]*inv_transf[0,:,1]  # the partial_{21}
                H_v[i,:,2] = HB[i,:,0]*inv_transf[1,:,0] + HB[i,:,2]*inv_transf[1,:,1]  # the partial_{12}
                H_v[i,:,3] = HB[i,:,1]*inv_transf[1,:,0] + HB[i,:,3]*inv_transf[1,:,1]  # the partial_{12}
            '''
        else:
            H_v = 0

        return H_v

    def fem_functions_value_ref(self, elem_index, f_h, x):
        """
        Remark: This part is highly finite element relative!!!
        This method is used to compute the value of a finite element function at the reference point x.
        elem_index can be '[]' or an index array(1D) used to specific the element.
        f_h is a 1D array. 1-by-dim
        The output is a 1-by-Nt array.
        """
        # ---------------------------------------------------------------------------
        # the finite element
        fem = self.space_fem
        # the mesh
        mesh = self.space_mesh
        # get the free_deg
        free_deg_all = self.dof_management()
        # find the elements
        if len(elem_index) == 0:
            free_deg = free_deg_all
        else:
            free_deg = free_deg_all[elem_index, :]

        # the function value at the degrees of freedom : Nt-by-3
        m_fh = f_h[free_deg - 1]
        # compute the value of basis functions at the reference point: Nt-by-3
        B_v = self.fem_basis_functions_value_ref(elem_index, x)
        # compute the values of f_h on each element at the reference point
        v = np.sum(m_fh * B_v, axis=1)

        return v

    def fem_functions_gradient_ref(self, elem_index, f_h, x):
        """
        Remark: This part is highly finite element relative!!!
        This method is used to compute the gradient of a finite element function at the reference point x.
        elem_index can be '[]' or an index array(1D) used to specific the element.
        f_h is a 1D array. 1-by-dim
        The output is a Nt-by-2 array.
        """
        # ---------------------------------------------------------------------------
        # the finite element
        fem = self.space_fem
        # the mesh
        mesh = self.space_mesh
        # get the free_deg
        free_deg_all = self.dof_management()
        # find the elements
        if len(elem_index) == 0:
            free_deg = free_deg_all
        else:
            free_deg = free_deg_all[elem_index, :]

        # the function value at the degrees of freedom : Nt-by-3
        m_fh = f_h[free_deg - 1]
        # compute the gradient of basis functions at the reference point: 3-by-Nt-by-2
        G_v = self.fem_basis_functions_gradient_ref(elem_index, x)
        G_v = np.transpose(G_v, (2, 1, 0))  # 2-by-Nt-by-3
        # compute the gradients of f_h on each element at the reference point
        Nt = free_deg.shape[0]
        gv = np.zeros((Nt, 2))
        for i in range(2):
            gv[:, i] = np.sum(m_fh * G_v[i, :, :], axis=1)

        return gv

    def fem_functions_hessian_ref(self, elem_index, f_h, x):
        """
        Remark: This part is highly finite element relative!!!
        This method is used to compute the Hessian of a finite element function at the reference point x.
        elem_index can be '[]' or an index array(1D) used to specific the element.
        f_h is a 1D array. 1-by-dim
        The output is a Nt-by-4 array.
        """
        # ---------------------------------------------------------------------------
        # the finite element
        fem = self.space_fem
        # the mesh
        mesh = self.space_mesh
        # get the free_deg
        free_deg_all = self.dof_management()
        # find the elements
        if len(elem_index) == 0:
            free_deg = free_deg_all
        else:
            free_deg = free_deg_all[elem_index, :]

        # the function value at the degrees of freedom : Nt-by-3
        m_fh = f_h[free_deg - 1]
        # compute the Hessian of basis functions at the reference point: 3-by-Nt-by-4
        H_v = self.fem_basis_functions_hessian_ref(elem_index, x)
        H_v = np.transpose(H_v, (2, 1, 0))  # 4-by-Nt-by-3
        # compute the gradients of f_h on each element at the reference point
        Nt = free_deg.shape[0]
        hv = np.zeros((Nt, 4))
        for i in range(4):
            hv[:, i] = np.sum(m_fh * H_v[i, :, :], axis=1)

        return hv

        # -----------------------------------------------------------
    def get_freeDof(self):
        dim = self.space_dim
        Nn = self.space_mesh.node.shape[0]
        Ne = self.space_mesh.nd4ed.shape[0]
        bd_ed = np.hstack((self.space_mesh.dirichlet_ed.transpose().squeeze(), self.space_mesh.neumann_ed.transpose().squeeze())).astype(int)
        free_base = np.hstack((np.kron(Nn * np.arange(self.space_fem.dof[0]), np.ones((1, len(bd_ed)))),
                            np.kron(Nn * self.space_fem.dof[0] + Ne * np.arange(self.space_fem.dof[1]),
                                    np.ones((1, len(bd_ed))))))
        un_freedom = free_base + np.hstack((np.kron(np.ones((1, self.space_fem.dof[0])), self.space_mesh.nd4ed[bd_ed - 1, 0]),
                                            np.kron(np.ones((1, self.space_fem.dof[1])), bd_ed)))
        un_freedom = un_freedom.astype(int)
        # the free dofs
        return np.setdiff1d(np.arange(dim, dtype=int) + 1, un_freedom[0])-1
    # This part is for the cases where the points x are not related to a reference point.
    # Each given element has one and only one point.
    # This is useful when one needs to deal with edges and nested meshes related computing.

    def fem_basis_functions_value(self, elem_index, x):
        pass

    def fem_basis_functions_gradient(self, elem_index, x):
        pass

    def fem_basis_functions_hessian(self, elem_index, x):
        pass

    def fem_functions_value(self, elem_index, f_h, x):
        pass

    def fem_functions_gradient(self, elem_index, f_h, x):
        pass

    def fem_functions_hessian(self, elem_index, f_h, x):
        pass


"""
# Test
# the coarse partition
coarse_partition = CoarsePartitionDirichlet()
# the mesh container
mesh0 = Mesh()
# the initial mesh
mesh0 = initial_mesh(coarse_partition, mesh0)
# the finite element
FEM = FiniteElementLP1('LP1')
# the finite element space
Vh = FESpace(FEM, mesh0)
# degrees of freedom
# Free_dof = Vh.dof_management()
# print(Free_dof)
n = Vh.space_dim
f_h = np.ones((1, n))[0]
x = np.array([0, 1])
v = Vh.fem_functions_value_ref([], f_h, x)
print('The values of f_h at x is :')
print(v)
gv = Vh.fem_functions_gradient_ref([], f_h, x)
print('The gradients of f_h at x is :')
print(gv)
hv = Vh.fem_functions_hessian_ref([], f_h, x)
print('The Hessian of f_h at x is :')
print(hv)
"""
