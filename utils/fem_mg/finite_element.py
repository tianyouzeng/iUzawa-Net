# This part defines the finite element

# Copyright @ Zhiyu Tan-Nov. 11, 2022.
# ==================================================================================
import numpy as np


class FiniteElement:
    """
    This is the finite element class, which includes the basic information of a finite element.
    """

    def __init__(self, name):
        # the name of the finite element
        self.name = name  # np.array([], dtype=str)
        # the type of the element domain
        self.K_type = np.array([], dtype=str)
        # the reference element
        self.K_reference = np.array([], dtype=float)
        # the degrees of freedom: 1-by-3
        self.dof = np.array([], dtype=int)
        # the total local dofs : 1-by-1
        self.total_local_dofs = np.array([], dtype=int)

    def printname(self):
        print('This is the %s finite element.\n' % self.name)


class FiniteElementLP1(FiniteElement):
    """
    This class gives the definition of the Lagrange P1 finite element.
    """

    def __init__(self, name):
        # FiniteElement.__init__(self, name) # keep inheritance of the parent's class FiniteElement
        super().__init__(name)
        # self.name = 'LP1'
        self.K_type = 'Triangle'
        self.K_reference = np.array(
            [[0, 1, 0], [0, 0, 1]])  # vertices of the reference element domain, counterclockwise
        self.dof = np.array([1, 0, 0])  # each [vertex, edge, element]
        # the total local dofs
        self.total_local_dofs = 3

    @staticmethod
    def basis_functions_value(x):
        """
        This method returns the value of the basis functions at a reference point x.
        x is a 1-by-2 array.
        The output is a 1-by-3 array.
        """
        return np.array([np.hstack((1 - x[0] - x[1], x[0], x[1]))])

    @staticmethod
    def basis_functions_gradient(x):
        """
        This method returns the gradient of the basis functions at a reference point x.
        x is a 1-by-2 array.
        The output is a 2-by-3 array.
        """
        return np.array([[-1, 1, 0], [-1, 0, 1]])

    @staticmethod
    def basis_functions_hessian(x):
        """
        This method returns the Hessian of the basis functions at a reference point x.
        x is a 1-by-2 array.
        The output is a 4-by-3 array.
        """
        return np.zeros((4, 3))

        # --------------------------------------------------------------------------------

    @staticmethod
    def basis_functions_value_ve(x):
        """
        This method returns the value of the basis functions at some reference points x.
        x is a 2-by-Nt array.
        The output is a 3-by-Nt array. (total_local_dof-by-Nt)
        """
        return np.vstack((1 - x[0] - x[1], x[0], x[1]))

    @staticmethod
    def basis_functions_gradient_ve(x):
        """
        This method returns the gradient of the basis functions at some reference points x.
        x is a 2-by-Nt array.
        The output is a 2-by-3-by-Nt array. (2-by-total_local_dof-by-Nt)
        """
        # the number of the given points
        n = x.shape[1]
        # partial x
        gx = np.array([[-1, 1, 0]]).T @ np.ones((1, n))
        # partial y
        gy = np.array([[-1, 0, 1]]).T @ np.ones((1, n))
        return np.array([gx, gy])

    @staticmethod
    def basis_functions_hessian_ve(x):
        """
        This method returns the Hessian of the basis functions at some reference points x.
        x is a 2-by-Nt array.
        The output is a 4-by-3-by-Nt array. (4-by-total_local_dof-by-Nt)
        """
        # the number of the given points
        n = x.shape[1]
        return np.zeros((4, 3, n))


# debug
# FEM = FiniteElementLP1('LP1')
# FEM.printname()
