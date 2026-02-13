# This subroutine is used to define the classes and functions related to the mesh.
# This is only for triangulation with triangles.
# Copyright @ Zhiyu Tan-Nov. 11, 2022.
# ==================================================================================
# initial coarse partition, mesh, initial mesh, refine mesh
# the initial coarse partition

import numpy as np
from scipy.sparse import csr_matrix, find, triu

class CoarsePartition:
    """
    This CoarsePartition class gives an initial coarse partition of the domain
    with only the crucial part of the mesh.
    """
    # mesh type
    # mesh_type = 'Triangle'
    # # nodes Nn-by-2
    # node = np.array(([1 / 2, 1 / 2], [0, 0], [1, 0], [1, 1], [0, 1]), dtype='float')  # node.transpose()
    # # elements NT-by-3
    # elem = np.array(([1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 2]), dtype='int')
    # # nodes related to the dirichlet boundary condition: Ne_bd-by-2
    # dirichlet = np.array(([2, 3], [3, 4], [4, 5], [5, 2]), dtype='int')
    # # nodes related to the neumann boundary condition
    # neumann = np.array([], dtype='int')
    # # nodes at corners of the domain: 1-by-Nn_c
    # corner = np.array([2, 3, 4, 5], dtype='int')

    mesh_type = 'Triangle'
    # nodes Nn-by-2
    node = np.array(([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]), dtype='float')  # node.transpose()
    # elements NT-by-3
    elem = np.array(([3, 4, 1], [2, 1, 4]), dtype='int')
    # nodes related to the dirichlet boundary condition: Ne_bd-by-2
    dirichlet = np.array([], dtype='int')
    # nodes related to the neumann boundary condition
    neumann = np.array([], dtype='int')
    # nodes at corners of the domain: 1-by-Nn_c
    corner = np.array([1, 3, 4, 2], dtype='int')


class CoarsePartitionDirichlet(CoarsePartition):
    """
    This CoarsePartition class gives an initial coarse partition of the domain
    with only the crucial part of the mesh.
    """
    # nodes related to the dirichlet boundary condition: Ne_bd-by-2
    dirichlet = np.array(([1, 3], [3, 4], [4, 2], [2, 1]), dtype='int')


class CoarsePartitionNeumann(CoarsePartition):
    """
    This CoarsePartition class gives an initial coarse partition of the domain
    with only the crucial part of the mesh.
    """
    pass


class Mesh:
    """
    The Mesh class defines the basic structures of a mesh.
    It will be used as a container.
    """

    def __init__(self):
        """
        # mesh type: triangle, quadrilateral or polygon
        self.mesh_type = []
        # nodes : Nn-by-2
        self.node = []
        # elements : NT-by-3, NT-by-4, ...
        self.elem = []
        # elements to edges : NT-by-3, NT-by-4, ...
        self.edge = []
        # nodes index related to the corner of the domain: 1-by-Nn_c
        self.corner = []
        # nodes to edges: Ne-by-2
        self.nd4ed = []
        # nodes related to the dirichlet boundary condition : Nnb_d-by-2
        self.dirichlet_nd = []
        # nodes related to the neumann boundary condition : Nnb_n-by-2
        self.neumann_nd = []
        # boundary edges related to the dirichlet boundary condition: Neb_d-by-1
        self.dirichlet_ed = []
        # boundary edges related to the neumann boundary condition: Neb_n-by-1
        self.neumann_ed = []
        # The parent of the element in the previous level mesh: NT-by-1
        self.parent4el = []
        # The parent of the edge in the previous level mesh: Ne-by-1
        self.parent4el = []
        # The subordinates of each element in next level. NT-by-4, ...
        self.sub4el = []
        # The subordinates of each edge in next level. Ne-by-2, ...
        self.sub4ed = []
        """

        # mesh type: triangle, quadrilateral or polygon
        self.mesh_type = np.array([], dtype=str)
        # nodes : Nn-by-2
        self.node = np.array([], dtype='float')
        # elements : NT-by-3, NT-by-4, ...
        self.elem = np.array([], dtype='int')
        # elements to edges : NT-by-3, NT-by-4, ...
        self.edge = np.array([], dtype='int')
        # nodes index related to the corner of the domain: 1-by-Nn_c
        self.corner = np.array([], dtype='int')
        # nodes to edges: Ne-by-2
        self.nd4ed = np.array([], dtype='int')
        # nodes related to the dirichlet boundary condition : Nnb_d-by-2
        self.dirichlet_nd = np.array([], dtype='int')
        # nodes related to the neumann boundary condition : Nnb_n-by-2
        self.neumann_nd = np.array([], dtype='int')
        # boundary edges related to the dirichlet boundary condition: Neb_d-by-1
        self.dirichlet_ed = np.array([], dtype='int')
        # boundary edges related to the neumann boundary condition: Neb_n-by-1
        self.neumann_ed = np.array([], dtype='int')
        # The parent of the element in the previous level mesh: NT-by-1
        self.parent4el = csr_matrix(np.array([], dtype='int'))
        # The parent of the edge in the previous level mesh: Ne-by-1
        self.parent4ed = csr_matrix(np.array([], dtype='int'))
        # The subordinates of each element in next level. NT-by-4, ...
        self.sub4el = np.array([], dtype='int')
        # The subordinates of each edge in next level. Ne-by-2, ...
        self.sub4ed = np.array([], dtype='int')

    def transform_information(self):
        """
                This method gives the transfer information between the physical elements and the reference elements.
                It will also return the area of each element.
                The output is:
                          area : a NT-by-1 array, which is given in 2D array form, |T|
                        transf : a 2-by-NT-by-2 array, K --> T
                    inv_transf : a 2-by-NT-by-2 array, T --> K
                """
        # the transfer matrix
        transf1 = self.node[self.elem[:, 1] - 1, :] - self.node[self.elem[:, 0] - 1, :]
        transf2 = self.node[self.elem[:, 2] - 1, :] - self.node[self.elem[:, 0] - 1, :]
        transf = np.array([transf1, transf2])
        # the area
        area = 0.5 * (transf[0, :, 0] * transf[1, :, 1] - transf[0, :, 1] * transf[1, :, 0])
        # the inverse of the transfer matrix
        inv_transf00 = 0.5 * transf[1, :, 1] / area  # inv_transf[0,:,0]
        inv_transf10 = -0.5 * transf[1, :, 0] / area  # inv_transf[1,:,0]
        inv_transf01 = -0.5 * transf[0, :, 1] / area  # inv_transf[0,:,1]
        inv_transf11 = 0.5 * transf[0, :, 0] / area  # inv_transf[1,:,1]
        inv_transf0 = np.vstack((inv_transf00, inv_transf01)).T
        inv_transf1 = np.vstack((inv_transf10, inv_transf11)).T
        inv_transf = np.array([inv_transf0, inv_transf1])
        # print('The output of transform_information is in the order: area, inv_transf, transf:\n')
        return np.array([area]).T,  inv_transf,  transf
        # return {'array': np.array([area]).T, 'inv_transf': inv_transf, 'transf': transf}

    def edge_length(self):
        """
                This method compute the length of edges of the mesh.
                The output is:
                       le : a Ne-by-1 array, which in 2D array form |e|
                """
        print('The length of edges is:\n ')
        le = np.float_power(np.sum((self.node[self.nd4ed[:, 1] - 1, :] - self.node[self.nd4ed[:, 0] - 1, :]) ** 2, 1),
                            0.5)
        return np.array([le])

    def mesh_plot(self):
        pass

    def label_mesh(self):
        pass

    def edges_plot(self, edges):
        pass


# the initial mesh
# coarse_partition = CoarsePartition()
# mesh0 = Mesh()


def initial_mesh(coarse_partition: CoarsePartition, mesh0: Mesh):
    """
    This subroutine is used to generate a mesh from a partition.
    param coarse_partition: a CoarsePartition instance
    :param mesh0: an (empty) Mesh instance
    :return: mesh0:
    """
    # start of the function
    # print('Generate the mesh data:\n')
    # ---------------------------------------
    # reload the data
    node = coarse_partition.node
    elem = coarse_partition.elem
    dirichlet = coarse_partition.dirichlet
    neumann = coarse_partition.neumann
    corner = coarse_partition.corner
    # ---------------------------------------
    # the number of the nodes
    Nn = node.shape[0]
    # the number of the elements
    NT = elem.shape[0]
    # ---------------------------------------
    # numerate the edges
    # ---------------------------------------
    # generate the adjacent matrix of the graph related to the partition
    index_b = [1, 2, 0]
    # Note: index starting from 0 in Python.
    graph_matrix_dir = csr_matrix(
        (np.ones([NT, 3], dtype='int').reshape(-1), ((elem - 1).reshape(-1), (elem[:, index_b] - 1).reshape(-1))),
        shape=(Nn, Nn))
    graph_matrix = graph_matrix_dir + graph_matrix_dir.T  # without direction
    # index of the nonzero items
    I = find(triu(graph_matrix))
    # the edges with endpoints: Ne-by-2
    nd4ed = (np.array([I[0], I[1]]) + 1).transpose()
    # adjust the direction of the boundary edges: counterclockwise
    # Note: matrix type can't be used as index
    index_of_edge_unmatched = np.asarray(graph_matrix_dir[I[0], I[1]] == 0)
    # change the direction
    nd4ed[index_of_edge_unmatched[0], :] = nd4ed[index_of_edge_unmatched[0], ::-1]
    # ---------------------------------------
    # the number of edges
    Ne = nd4ed.shape[0]
    # generate the nodes for edges matrix
    edge4node = csr_matrix((np.arange(Ne, dtype='int') + 1, (I[0], I[1])), shape=(Nn, Nn))
    edge4node = edge4node + edge4node.T
    # ---------------------------------------
    # The Dirichlet boundary
    dirichlet_ed = np.array([], dtype='int')
    Ne_bd = len(dirichlet)
    if Ne_bd > 0:
        dirichlet_ed = np.asarray(
            edge4node[dirichlet[:, 0] - 1, dirichlet[:, 1] - 1]).transpose()  # convert to an array

    # The Neumann boundary
    neumann_ed = np.array([], dtype='int')
    if hasattr(coarse_partition, 'neumann'):
        Ne_bn = len(neumann)
        if Ne_bn > 0:
            neumann_ed = np.asarray(edge4node[neumann[:, 0] - 1, neumann[:, 1] - 1]).transpose()  # convert to an array

    # ---------------------------------------
    # the matrix of the representation of the topology of nodes and the edges in each element
    index_b2 = [2, 0, 1]
    edge = np.asarray(edge4node[(elem[:, index_b] - 1).reshape(-1), (elem[:, index_b2] - 1).reshape(-1)]).reshape(
        (NT, 3))
    # ---------------------------------------
    # update the mesh
    mesh0.mesh_type = np.array(coarse_partition.mesh_type, dtype=str)
    mesh0.node = node
    mesh0.elem = elem
    mesh0.edge = edge
    mesh0.corner = corner
    mesh0.nd4ed = nd4ed
    mesh0.dirichlet_nd = dirichlet
    mesh0.dirichlet_ed = dirichlet_ed
    mesh0.neumann_nd = neumann
    mesh0.neumann_ed = neumann_ed
    # ------------------------------------------
    #print('The initial mesh has been generated.\n')

    return mesh0


# refine the mesh

# --------------------------------------------------------------------------
# Some Remarks:
#  1. The elements are divided into 4 triangles based on the middle point
#     of each edge.
#         1                       1
#        / \                     / \
#       /   \      ---\         /   \
#      /     \     ---/        6-----5
#     /       \               / \   / \
#    /         \             /   \ /   \
#   2-----------3           2-----4-----3
#  2. The local index of the new sub-triangles are based on the vertex it
#     shares with its owner.
#         1                       1
#        / \                     / \
#       /   \      ---\         / 1 \
#      /     \     ---/        /-----\  1.Local_elem_number = local_node_number
#     /       \               / \ 4 / \ 2.The center one is the same as the
#    /         \             / 2 \ / 3 \  owner.
#   2-----------3           2-----------3
#  3. The new nodes are numbered based on the edge to which it belongs.
#         1                       1
#        / \                     / \
#       /   \      ---\         / 1 \
#      /     \     ---/        6-----5     N_node = Nn + edge_number
#     /       \               / \ 4 / \
#    /         \             / 2 \ / 3 \
#   2-----------3           2-----4-----3
#  4. The new edges are numbered based on the new elements and new nodes.
#  5. In such way, the first NT nodes is the nodes from the old mesh.
# --------------------------------------------------------------------------

def refine_mesh_tri(old_mesh: Mesh) -> tuple[Mesh, Mesh] | None:
    """
    This subroutine is used to refine the mesh uniformly.
    This is only for triangles.
    """
    # --------------------------------------------------------
    # check the mesh type
    if np.any(old_mesh.mesh_type != 'Triangle'):
        print('The input mesh is not a simplicial triangulation. \n')
        print('Please give a simplicial triangulation or use the proper refinement subroutine.\n')
        return None
    # else:
    #    print('The given mesh is a simplicial mesh. We refine it uniformly now.\n')

    # print('The given mesh is a simplicial mesh. We refine it uniformly now.\n')
    # --------------------------------------------------------
    # generate the new mesh container
    new_mesh = Mesh()
    # --------------------------------------------------------
    # the basic information of the given mesh
    Nn = old_mesh.node.shape[0]  # the number of the nodes
    NT = old_mesh.elem.shape[0]  # the number of the elements
    Ne = old_mesh.nd4ed.shape[0]  # the number of the edges
    # --------------------------------------------------------
    # compute the midpoint of the edges
    mid_node = 1 / 2 * (old_mesh.node[old_mesh.nd4ed[:, 0] - 1, :] + old_mesh.node[old_mesh.nd4ed[:, 1] - 1, :])
    # --------------------------------------------------------
    #print('Refine the mesh: deal with the nodes\n')
    # the number of nodes in the new mesh
    new_Nn = Nn + Ne
    # new nodes
    new_mesh.node = np.append(old_mesh.node, mid_node, axis=0)
    # deal with the boundary nodes
    if len(old_mesh.dirichlet_ed) > 0:
        # dn1 = np.array([old_mesh.dirichlet_nd[:,0], Nn + old_mesh.dirichlet_ed.T[0,:]]).T
        # dn2 = np.array([Nn + old_mesh.dirichlet_ed.T[0,:], old_mesh.dirichlet_nd[:,1]]).T
        # new_mesh.dirichlet_nd = np.append(dn1, dn2, axis = 0)
        # another way
        Ne_bd = old_mesh.dirichlet_ed.shape[0]
        w1 = [old_mesh.dirichlet_nd[:, 0], Nn + old_mesh.dirichlet_ed.T[0, :], Nn + old_mesh.dirichlet_ed.T[0, :], old_mesh.dirichlet_nd[:, 1]]
        new_mesh.dirichlet_nd = np.array(w1).reshape(2, 2 * Ne_bd).T
        # new_mesh.dirichlet_nd = np.array([old_mesh.dirichlet_nd[:, 0], Nn + old_mesh.dirichlet_ed.T[0, :], \
        #                                  Nn + old_mesh.dirichlet_ed.T[0, :], old_mesh.dirichlet_nd[:, 1]]).reshape(2,2 * Ne_bd).T

    if len(old_mesh.neumann_ed) > 0:
        # nn1 = np.array([old_mesh.neumann_nd[:,0], Nn + old_mesh.neumann_ed.T[0,:]]).T
        # nn2 = np.array([Nn + old_mesh.neumann_ed.T[0,:], old_mesh.neumann_nd[:,1]]).T
        # new_mesh.neumann_nd = np.append(dn1, dn2, axis = 0)
        # another way
        Ne_bn = old_mesh.neumann_ed.shape[0]
        new_mesh.neumann_nd = np.array([old_mesh.neumann_nd[:, 0], Nn + old_mesh.neumann_ed.T[0, :], \
                                          Nn + old_mesh.neumann_ed.T[0, :], old_mesh.neumann_nd[:, 1]]).reshape(2,
                                                                                                                2 * Ne_bn).T

    # --------------------------------------------------------
    #print('Refine the mesh: deal with the elements\n')
    # the number of the elements of the mesh
    new_NT = 4 * NT
    # the new nodes on the edges of the parent elements
    N_nd4el = Nn + old_mesh.edge
    # generate the new elements
    elem1 = np.array([old_mesh.elem[:, 0], N_nd4el[:, 2], N_nd4el[:, 1]])
    elem2 = np.array([N_nd4el[:, 2], old_mesh.elem[:, 1], N_nd4el[:, 0]])
    elem3 = np.array([N_nd4el[:, 1], N_nd4el[:, 0], old_mesh.elem[:, 2], ])
    elem4 = N_nd4el.T
    new_mesh.elem = np.hstack((elem1, elem2, elem3, elem4)).reshape(3, new_NT).T
    # --------------------------------------------------------
    #print('Refine the mesh: deal with the edges\n')
    # --------------------------------------------------------
    # numerate the edges
    # --------------------------------------------------------
    # generate the adjacent matrix of the graph related to the partition
    index_b = [1, 2, 0]
    # Note: index starting from 0 in Python.
    elem = new_mesh.elem
    graph_matrix_dir = csr_matrix(
        (np.ones([new_NT, 3], dtype=int).reshape(-1), ((elem - 1).reshape(-1), (elem[:, index_b] - 1).reshape(-1))),
        shape=(new_Nn, new_Nn))
    graph_matrix = graph_matrix_dir + graph_matrix_dir.T  # without direction
    # index of the nonzero items
    I = find(triu(graph_matrix))
    # the edges with endpoints: Ne-by-2
    nd4ed = (np.array([I[0], I[1]]) + 1).transpose()
    # adjust the direction of the boundary edges: counterclockwise
    # Note: matrix type can't be used as index
    index_of_edge_unmatched = np.asarray(graph_matrix_dir[I[0], I[1]] == 0)
    # change the direction
    nd4ed[index_of_edge_unmatched[0], :] = nd4ed[index_of_edge_unmatched[0], ::-1]
    # the new nodes for edges array
    new_mesh.nd4ed = nd4ed
    # --------------------------------------------------------
    #print('Refine the mesh: deal with the boundary edges\n')
    # the number of the edges in the new mesh
    new_Ne = 2 * Ne + 3 * NT
    # generate the nodes for edges matrix
    edge4node = csr_matrix((np.arange(new_Ne, dtype=int) + 1, (I[0], I[1])), shape=(new_Nn, new_Nn))
    edge4node = edge4node + edge4node.T
    # The Dirichlet boundary
    Ne_bd = len(old_mesh.dirichlet_nd)
    if Ne_bd > 0:
        new_mesh.dirichlet_ed = np.asarray(edge4node[new_mesh.dirichlet_nd[:, 0] - 1, new_mesh.dirichlet_nd[:, 1] - 1]).transpose()  # convert to an array

    # The Neumann boundary
    Ne_bn = len(old_mesh.neumann_nd)
    if Ne_bn > 0:
        new_mesh.neumann_ed = np.asarray(
            edge4node[new_mesh.neumann_nd[:, 0] - 1, new_mesh.neumann_nd[:, 1] - 1]).transpose()  # convert to an array

    # --------------------------------------------------------
    #print('Refine the mesh: the relationship between new edges and new elements.\n')
    # the matrix of the representation of the topology of nodes and the edges in each element
    index_b2 = [2, 0, 1]
    new_mesh.edge = np.asarray(
        edge4node[(elem[:, index_b] - 1).reshape(-1), (elem[:, index_b2] - 1).reshape(-1)]).reshape((new_NT, 3))
    # --------------------------------------------------------
    # The relationship between the new mesh and the old mesh
    #print('Refine the mesh: the relationship between the new mesh and the old mesh.\n')
    # subordinates of the old elements
    old_sub4el = np.array([np.arange(NT) + 1]).T @ np.ones((1, 4)) + np.ones((NT, 1)) @ np.array(
        [[0, NT, 2 * NT, 3 * NT]])
    # parents of the new elements
    new_parent4el = csr_matrix(((np.array([np.arange(NT) + 1]).T @ np.ones((1, 4))).reshape(-1),
                                ((old_sub4el - 1).reshape(-1), (np.ones((4 * NT, 1)) - 1).reshape(-1))),
                               shape=(4 * NT, 1))
    # subordinates of the old edges
    old_sub4ed = np.vstack((np.asarray(edge4node[old_mesh.nd4ed[:, 0] - 1, Nn + np.arange(Ne)]),
                            np.asarray(edge4node[Nn + np.arange(Ne), old_mesh.nd4ed[:, 1] - 1]))).transpose()
    # parents of the new edges
    new_parent4ed = csr_matrix(((np.array([np.arange(Ne) + 1]).T @ np.ones((1, 2))).reshape(-1),
                                ((old_sub4ed - 1).reshape(-1), (np.ones((2 * Ne, 1)) - 1).reshape(-1))),
                               shape=(new_Ne, 1))
    # --------------------------------------------------------
    # update: old_mesh
    old_mesh.sub4el = old_sub4el
    old_mesh.sub4ed = old_sub4ed
    # new mesh
    new_mesh.parent4el = new_parent4el
    new_mesh.parent4ed = new_parent4ed
    # ---------------------------------------------------------
    #print('The uniform refinement has been finished.\n')
    return new_mesh, old_mesh
# Test
# the coarse partition
# coarse_partition = CoarsePartition()
# the mesh container
# mesh0 = Mesh()
# the initial mesh
# mesh0 = initial_mesh(coarse_partition, mesh0)
# w1 = mesh0.transform_information()[1]
# print(w1)
# w2 = mesh0.edge_length()
# print(w2)
# refine the mesh
# new_mesh, old_mesh = refine_mesh_tri(mesh0)
# print(new_mesh.dirichlet_nd)
# print(new_mesh.node)
# print(new_mesh.elem)
