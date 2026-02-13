from .mesh import refine_mesh_tri
from .fe_space import FESpace
import scipy as scp
import numpy as np
from scipy.sparse.linalg import spsolve_triangular


def Res(old_mesh,new_mesh):
    ii1=np.arange(0, len(old_mesh.node))
    ii2=np.arange(len(old_mesh.node), len(new_mesh.node))
    ii3=np.arange(len(old_mesh.node), len(new_mesh.node))
    ii=np.hstack((ii1,ii2,ii3))
    jj1=np.arange(0,len(old_mesh.node))
    jj2=old_mesh.nd4ed[:,0]-1
    jj3=old_mesh.nd4ed[:,1]-1
    jj=np.hstack((jj1,jj2,jj3))
    jj=np.hstack((jj1,jj2,jj3))
    ss1=np.ones(len(old_mesh.node))
    ss2=0.5*np.ones(len(new_mesh.node)-len(old_mesh.node))
    ss3=0.5*np.ones(len(new_mesh.node)-len(old_mesh.node))
    ss=np.hstack((ss1,ss2,ss3))
    P=scp.sparse.csr_matrix((ss, (ii,jj)),shape=(len(new_mesh.node),len(old_mesh.node)))
    return P


def rearrange_by_nodes(func, nodes):
    """
    Rearranges a 2D array 'f' into a 1D array based on the order of 'nodes'.

    Args:
        f (np.ndarray): An N*N NumPy array representing a function on a uniform grid.
        nodes (np.ndarray): An (N*N, 2) NumPy array of permuted grid coordinates.

    Returns:
        np.ndarray: A 1D NumPy array of size N*N with rearranged values.
    """
    N = func.shape[0]

    # Separate the x and y coordinates from the nodes array
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]

    # Convert the physical coordinates to integer array indices.
    # We use np.round to handle potential floating-point inaccuracies.
    # The `astype(np.intp)` is used to ensure the indices are of the correct integer type.
    col_indices = np.round(x_coords * (N - 1)).astype(np.intp)
    row_indices = np.round(y_coords * (N - 1)).astype(np.intp)

    # Use advanced indexing to get the values from f in the desired order.
    # f[i, j] corresponds to the coordinate (x[j], y[i]), so we use
    # row_indices for the first axis and col_indices for the second.
    f_rearranged = func[row_indices, col_indices]

    return f_rearranged.reshape(-1, 1)

def rearrange_by_nodes_inverse(func, nodes):
    """
    Rearranges a 1D array 'func' back into a 2D array based on the order of 'nodes'.
    This is the inverse operation of 'rearrange_by_nodes'.

    Args:
        func (np.ndarray): A 1D NumPy array or (N*N, 1) array of size N*N.
        nodes (np.ndarray): An (N*N, 2) NumPy array of permuted grid coordinates.

    Returns:
        np.ndarray: An N*N NumPy array representing a function on a uniform grid.
    """
    num_nodes = nodes.shape[0]
    N = int(np.round(np.sqrt(num_nodes)))
    
    if N * N != num_nodes:
        raise ValueError("The number of nodes must be a perfect square.")

    # Ensure func is 1D for indexing
    func_flat = func.flatten()

    if func_flat.shape[0] != num_nodes:
        raise ValueError(f"Input function size {func_flat.shape[0]} does not match number of nodes {num_nodes}.")

    # Separate the x and y coordinates from the nodes array
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]

    # Convert the physical coordinates to integer array indices.
    col_indices = np.round(x_coords * (N - 1)).astype(np.intp)
    row_indices = np.round(y_coords * (N - 1)).astype(np.intp)

    result = np.zeros((N, N), dtype=func.dtype)
    result[row_indices, col_indices] = func_flat
    
    return result

def rearrange_by_nodes_batched(func, nodes):
    """
    Rearranges a 4D array 'func' of shape (batch_size, S, S, T) into a 2D array 
    of shape (batch_size, S*S*T) based on the order of 'nodes'.

    Args:
        func (np.ndarray): A (batch_size, S, S, T) NumPy array.
        nodes (np.ndarray): An (S*S, 2) NumPy array of permuted grid coordinates.

    Returns:
        np.ndarray: A 2D NumPy array of size (batch_size, S*S*T) with rearranged values.
    """
    S = func.shape[1]
    batch_size = func.shape[0]

    # Separate the x and y coordinates from the nodes array
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]

    # Convert the physical coordinates to integer array indices.
    col_indices = np.round(x_coords * (S - 1)).astype(np.intp)
    row_indices = np.round(y_coords * (S - 1)).astype(np.intp)

    # Use advanced indexing on the batched array.
    # This selects elements from each S x S slice for all batches and time steps.
    # The result has shape (batch_size, S*S, T).
    f_rearranged = func[:, row_indices, col_indices, :]

    # Reshape to the desired output shape (batch_size, S*S*T).
    return f_rearranged.transpose((0, 2, 1)).reshape(batch_size, -1)

def rearrange_by_nodes_batched_inverse(func, nodes):
    """
    Rearranges a 2D array 'func' of shape (batch_size, S*S*T) back to a 4D array 
    of shape (batch_size, S, S, T) based on the order of 'nodes'.

    Args:
        func (np.ndarray): A (batch_size, S*S*T) NumPy array.
        nodes (np.ndarray): An (S*S, 2) NumPy array of permuted grid coordinates.

    Returns:
        np.ndarray: A 4D NumPy array of size (batch_size, S, S, T) with values placed
                    according to the grid coordinates.
    """
    batch_size = func.shape[0]
    num_nodes = nodes.shape[0]
    S = int(np.sqrt(num_nodes))
    
    if S * S != num_nodes:
        raise ValueError("The number of nodes must be a perfect square.")
        
    T = func.shape[1] // num_nodes
    if num_nodes * T != func.shape[1]:
        raise ValueError("The length of the function array is not a multiple of the number of nodes.")

    func_reshaped = func.reshape(batch_size, T, num_nodes).transpose((0, 2, 1))
    
    # Create an output array to hold the grid-structured data
    result = np.zeros((batch_size, S, S, T), dtype=func.dtype)

    # Separate the x and y coordinates from the nodes array
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]

    # Convert the physical coordinates to integer array indices.
    col_indices = np.round(x_coords * (S - 1)).astype(np.intp)
    row_indices = np.round(y_coords * (S - 1)).astype(np.intp)

    # Use advanced indexing to place the values from func_reshaped into the result array.
    # We iterate through the batch dimension.
    for i in range(batch_size):
        result[i, row_indices, col_indices, :] = func_reshaped[i, :, :]

    return result


class MG:
    def __init__(self,mesh0,FEM,refinement_N):
        mesh_now=mesh0
        R_free_list=[]
        Vh_now = FESpace(FEM, mesh_now)
        Vh_list=[Vh_now]
        freeDOF_now=Vh_now.get_freeDof()
        freeDOF_list=[freeDOF_now]
        # new_mesh, old_mesh = refine_mesh_tri(mesh0)
        if refinement_N > 0:
            for i in range(refinement_N):
                mesh_tuple = refine_mesh_tri(mesh_now)
                assert mesh_tuple is not None, "Mesh refinement failed. Ensure the input mesh is a simplicial triangulation."
                mesh_now, mesh_old= mesh_tuple
                Vh_now = FESpace(FEM, mesh_now)
                Vh_list.append(Vh_now)
                freeDOF_c=freeDOF_now
                freeDOF_now=Vh_now.get_freeDof()
                freeDOF_list.append(freeDOF_now)
                R=Res(mesh_old,mesh_now)
                R_free_list.append(R[freeDOF_now , :][:, freeDOF_c ])
        n_grid=len(R_free_list)
        R_free_tr_list=[]
        mu0 = 2
        smoothingRatio = 2
        self.mui = [mu0]
        for i in range(n_grid):
            current_matrix =R_free_list[n_grid-1-i]
            R_free_tr_list.append(current_matrix)
            self.mui.append(self.mui[i] * smoothingRatio)
            if current_matrix.shape[1]<100:
                break
        self.Vh_list=Vh_list
        self.Vh=Vh_now
        self.freeDOF_list=freeDOF_list
        self.freeDOF=freeDOF_now
        self.R_free_list=R_free_tr_list
        self.level=len(R_free_tr_list)

    def set_V(self,A_now):
        A_free_list = [A_now]
        Smoother_list = [2 * A_now.diagonal().reshape(-1, 1)]
        
        for R_now in self.R_free_list:
            A_now = R_now.T @ (A_now @ R_now)
            Smoother_now = 2 * A_now.diagonal().reshape(-1, 1)
            A_free_list.append(A_now)
            Smoother_list.append(Smoother_now)
        
        invA_c = scp.linalg.inv(A_now.todense())
        self.A_free_list = A_free_list
        self.Smoother_list = Smoother_list
        self.invA_c = invA_c
        return A_free_list, Smoother_list, invA_c

    def V_fun(self, r, A_free_list=None, Smoother_list=None, invA_c=None):
        if A_free_list is not None:
            self.A_free_list = A_free_list
        if Smoother_list is not None:
            self.Smoother_list = Smoother_list
        if invA_c is not None:
            self.invA_c = invA_c    
        assert hasattr(self, 'A_free_list') and hasattr(self, 'Smoother_list') and hasattr(self, 'invA_c'), "set_V must be called before V_fun"
        # Initialize lists
        ri = [r]
        ei = []  # Prepare for level + 1 entries
        # First half of the calculation
        for j in range(self.level):
            ei.append(ri[j] / self.Smoother_list[j])  # Initial estimation
            for _ in range(self.mui[j] - 1):
                ei[j] += (ri[j] - self.A_free_list[j] @ ei[j]) /self.Smoother_list[j]
            ri.append(self.R_free_list[j].T @ (ri[j] - self.A_free_list[j] @ ei[j]))
        # Last entry for ei
        ei.append(self.invA_c @ ri[-1])
        # Second half of the calculation
        for k in range(self.level):
            i = self.level - k - 1
            ei[i] += self.R_free_list[i] @ ei[i + 1]
            for _ in range(self.mui[i]):
                ei[i] += (ri[i] - self.A_free_list[i] @ ei[i]) / self.Smoother_list[i]
        return ei[0]

    def W_fun(self, r):
        return self._w_cycle_recursive(0, r)

    def _w_cycle_recursive(self, level, r):
        if level == self.level:
            return self.invA_c @ r

        # Pre-smoothing
        e = np.zeros_like(r)
        for _ in range(self.mui[level]):
            e += (r - self.A_free_list[level] @ e) / self.Smoother_list[level]

        # Coarse-grid correction
        residual = self.R_free_list[level].T @ (r - self.A_free_list[level] @ e)
        
        e_coarse = self._w_cycle_recursive(level + 1, residual)
        if level + 1 < len(self.A_free_list):
             e_coarse += self._w_cycle_recursive(level + 1, residual - self.A_free_list[level+1] @ e_coarse)

        e += self.R_free_list[level] @ e_coarse

        # Post-smoothing
        for _ in range(self.mui[level]):
            e += (r - self.A_free_list[level] @ e) / self.Smoother_list[level]

        return e

class MG_Neumann:
    def __init__(self,mesh0,FEM,refinement_N):
        mesh_now=mesh0
        R_free_list=[]
        Vh_now = FESpace(FEM, mesh_now)
        Vh_list=[Vh_now]
        freeDOF_now=Vh_now.get_freeDof()
        freeDOF_list=[freeDOF_now]
        # new_mesh, old_mesh = refine_mesh_tri(mesh0)
        if refinement_N > 0:
            for i in range(refinement_N):
                mesh_tuple = refine_mesh_tri(mesh_now)
                assert mesh_tuple is not None, "Mesh refinement failed. Ensure the input mesh is a simplicial triangulation."
                mesh_now, mesh_old= mesh_tuple
                Vh_now = FESpace(FEM, mesh_now)
                Vh_list.append(Vh_now)
                freeDOF_c=freeDOF_now
                freeDOF_now=Vh_now.get_freeDof()
                freeDOF_list.append(freeDOF_now)
                R=Res(mesh_old,mesh_now)
                R_free_list.append(R[freeDOF_now , :][:, freeDOF_c ])
        n_grid=len(R_free_list)
        R_free_tr_list=[]
        mu0 = 2
        smoothingRatio = 2
        self.mui = [mu0]
        for i in range(n_grid):
            current_matrix =R_free_list[n_grid-1-i]
            R_free_tr_list.append(current_matrix)
            self.mui.append(self.mui[i] * smoothingRatio)
            if current_matrix.shape[1]<100:
                break
        self.Vh_list=Vh_list
        self.Vh=Vh_now
        self.freeDOF_list=freeDOF_list
        self.freeDOF=freeDOF_now
        self.R_free_list=R_free_tr_list
        self.level=len(R_free_tr_list)

    def set_V(self,A_now):
        A_free_list = [A_now]
        Smoother_list = [2 * A_now.diagonal().reshape(-1, 1)]
        
        for R_now in self.R_free_list:
            A_now = R_now.T @ (A_now @ R_now)
            Smoother_now = 2 * A_now.diagonal().reshape(-1, 1)
            A_free_list.append(A_now)
            Smoother_list.append(Smoother_now)
        
        invA_c = scp.linalg.inv(A_now.todense())
        self.A_free_list = A_free_list
        self.Smoother_list = Smoother_list
        self.invA_c = invA_c
        return A_free_list, Smoother_list, invA_c

    def V_fun(self, r, A_free_list=None, Smoother_list=None, invA_c=None):
        if A_free_list is not None:
            self.A_free_list = A_free_list
        if Smoother_list is not None:
            self.Smoother_list = Smoother_list
        if invA_c is not None:
            self.invA_c = invA_c    
        assert hasattr(self, 'A_free_list') and hasattr(self, 'Smoother_list') and hasattr(self, 'invA_c'), "set_V must be called before V_fun"
        # Initialize lists
        ri = [r]
        ei = []  # Prepare for level + 1 entries
        # First half of the calculation
        for j in range(self.level):
            ei.append(ri[j] / self.Smoother_list[j])  # Initial estimation
            for _ in range(self.mui[j] - 1):
                ei[j] += (ri[j] - self.A_free_list[j] @ ei[j]) /self.Smoother_list[j]
            ri.append(self.R_free_list[j].T @ (ri[j] - self.A_free_list[j] @ ei[j]))
        # Last entry for ei
        ei.append(self.invA_c @ ri[-1])
        # Second half of the calculation
        for k in range(self.level):
            i = self.level - k - 1
            ei[i] += self.R_free_list[i] @ ei[i + 1]
            for _ in range(self.mui[i]):
                ei[i] += (ri[i] - self.A_free_list[i] @ ei[i]) / self.Smoother_list[i]
        return ei[0]

    def W_fun(self, r):
        return self._w_cycle_recursive(0, r)

    def _w_cycle_recursive(self, level, r):
        if level == self.level:
            return self.invA_c @ r

        # Pre-smoothing
        e = np.zeros_like(r)
        for _ in range(self.mui[level]):
            e += (r - self.A_free_list[level] @ e) / self.Smoother_list[level]

        # Coarse-grid correction
        residual = self.R_free_list[level].T @ (r - self.A_free_list[level] @ e)
        
        e_coarse = self._w_cycle_recursive(level + 1, residual)
        if level + 1 < len(self.A_free_list):
             e_coarse += self._w_cycle_recursive(level + 1, residual - self.A_free_list[level+1] @ e_coarse)

        e += self.R_free_list[level] @ e_coarse

        # Post-smoothing
        for _ in range(self.mui[level]):
            e += (r - self.A_free_list[level] @ e) / self.Smoother_list[level]

        return e


# class MG_GS(MG):
#     def __init__(self, mesh0, FEM, refinement_N):
#         super().__init__(mesh0, FEM, refinement_N)

#     def set_V(self, A_now):
#         A_free_list = [A_now]
#         # For Gauss-Seidel, we need the lower triangular part of the matrix
#         Smoother_list = [scp.sparse.tril(A_now, format='csr')]
        
#         for R_now in self.R_free_list:
#             A_now = R_now.T @ (A_now @ R_now)
#             # For Gauss-Seidel, we need the lower triangular part of the matrix
#             Smoother_now = scp.sparse.tril(A_now, format='csr')
#             A_free_list.append(A_now)
#             Smoother_list.append(Smoother_now)
        
#         invA_c = scp.linalg.inv(A_now.todense())
#         return A_free_list, Smoother_list, invA_c

#     def V_fun(self, r, A_free_list, Smoother_list, invA_c):
#         # Initialize lists
#         ri = [r]
#         ei = [np.zeros_like(r)] * (self.level + 1)
        
#         # Pre-smoothing
#         for j in range(self.level):
#             e_now = np.zeros_like(ri[j])
#             for _ in range(self.mui[j]):
#                 U = scp.sparse.triu(A_free_list[j], k=1)
#                 e_now = spsolve_triangular(Smoother_list[j], ri[j] - U @ e_now, lower=True)
#             ei[j] = e_now
#             ri.append(self.R_free_list[j].T @ (ri[j] - A_free_list[j] @ ei[j]))
        
#         # Coarsest grid solver
#         ei[self.level] = invA_c @ ri[self.level]

#         # Post-smoothing
#         for j in range(self.level - 1, -1, -1):
#             ei[j] += self.R_free_list[j] @ ei[j + 1]
#             e_now = ei[j]
#             for _ in range(self.mui[j]):
#                 U = scp.sparse.triu(A_free_list[j], k=1)
#                 e_now = spsolve_triangular(Smoother_list[j], ri[j] - U @ e_now, lower=True)
#             ei[j] = e_now
            
#         return ei[0]

#     def W_fun(self, r, A_free_list, Smoother_list, invA_c):
#         return self._w_cycle_recursive(0, r, A_free_list, Smoother_list, invA_c)

#     def _w_cycle_recursive(self, level, r, A_free_list, Smoother_list, invA_c):
#         if level == self.level:
#             return invA_c @ r

#         # Pre-smoothing
#         e = np.zeros_like(r)
#         for _ in range(self.mui[level]):
#             U = scp.sparse.triu(A_free_list[level], k=1)
#             e = spsolve_triangular(Smoother_list[level], r - U @ e, lower=True)

#         # Coarse-grid correction
#         residual = self.R_free_list[level].T @ (r - A_free_list[level] @ e)
        
#         e_coarse = self._w_cycle_recursive(level + 1, residual, A_free_list, Smoother_list, invA_c)
#         if level + 1 < len(A_free_list):
#             e_coarse += self._w_cycle_recursive(level + 1, residual - A_free_list[level+1] @ e_coarse, A_free_list, Smoother_list, invA_c)

#         e += self.R_free_list[level] @ e_coarse

#         # Post-smoothing
#         for _ in range(self.mui[level]):
#             U = scp.sparse.triu(A_free_list[level], k=1)
#             e = spsolve_triangular(Smoother_list[level], r - U @ e, lower=True)

#         return e
