from scipy import sparse
from enum import Enum
from .geometry_util import *


DEFAULT_ANCHOR_WEIGHT = 1.0
flatten = lambda l: [item for sublist in l for item in sublist]


class LaplaceBeltramiWeighting(Enum):
    """ Weighting scheme to use for the Laplace-Beltrami operator.
        See Section 2.1 in Laplacian Mesh Processing (Olga Sorkine)
        Link: https://people.eecs.berkeley.edu/~jrs/meshpapers/Sorkine.pdf
    """
    UMBRELLA = 1
    COTANGENT_NO_AREA = 2
    COTANGENT_VORONOI_CELL = 3
    COTANGENT_ONE_THIRD_TRIANGLE = 4


class LaplacianBeltrami:
    """
    Holds Laplacian related information for a specific mesh.
    Note: assumes mesh connectivity doesn't change.
    """

    def __init__(self, mesh, weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA):
        """
        Constructs and calculates the LaplaceBeltrami operator for a specific mesh.
        :param mesh: 3d triangular mesh, in Trimesh format.
        :param weighting_scheme: Any of LaplaceBeltramiWeighting, determines how geometry is induced on LB.
        """
        self.mesh = mesh
        self.weighting_scheme = weighting_scheme

        self.sparse_mat_rows = None
        self.sparse_mat_cols = None
        self.sparse_mat_weights = None
        self.vertex_areas = None
        self.anchor_idx = []

        self._build_laplacian_matrix()

    def _build_laplacian_matrix(self):
        """
        :return: Initializes the data-structures required to build a sparse Laplacian matrix in coo format.
        """
        mesh = self.mesh
        vertex_degs = [len(neighbors) for neighbors in mesh.vertex_neighbors]
        sparse_mat_rows = [[row_idx] * (deg + 1) for row_idx, deg in enumerate(vertex_degs)]
        sparse_mat_cols = [neighbors + [row_idx] for row_idx, neighbors in enumerate(mesh.vertex_neighbors)]

        if self.weighting_scheme == LaplaceBeltramiWeighting.UMBRELLA:
            sparse_mat_weights = [[-1.0] * deg + [deg * 1.0] for deg in vertex_degs]
        else:
            vertex_areas = []
            edges_lut = joint_faces_graph(mesh)
            sparse_mat_weights = []
            for v_id, neighbors in enumerate(mesh.vertex_neighbors):
                v_weights = []
                voronoi_cell_components = []
                for neighbor_id in neighbors:
                    cots = []
                    for joint_vid in edges_lut.edges[v_id, neighbor_id]['facing_vertex']:
                        if joint_vid is None:  # Degenerated triangle
                            cots.append(0.0)
                            voronoi_cell_components.append(1e-12)
                        else:
                            e1 = get_vector(mesh, joint_vid, v_id)
                            e2 = get_vector(mesh, joint_vid, neighbor_id)
                            cot = np.dot(e1, e2) / np.linalg.norm(cross_product(e1, e2))
                            cots.append(cot)

                            if self.weighting_scheme == LaplaceBeltramiWeighting.COTANGENT_VORONOI_CELL:
                                kite_v1 = mesh.vertices[joint_vid]
                                kite_v2 = mean3(mesh, v_id, joint_vid, neighbor_id)
                                kite_v3 = mean2(mesh, v_id, joint_vid)
                                kite_v4 = mean2(mesh, v_id, neighbor_id)
                                kite_d1 = np.linalg.norm(kite_v1 - kite_v2)
                                kite_d2 = np.linalg.norm(kite_v3 - kite_v4)
                                voronoi_cell_component = kite_d1 * kite_d2 / 2.0
                                voronoi_cell_components.append(voronoi_cell_component)
                            elif self.weighting_scheme == LaplaceBeltramiWeighting.COTANGENT_ONE_THIRD_TRIANGLE:
                                e3 = get_vector(mesh, v_id, neighbor_id)
                                e4 = get_vector(mesh, v_id, joint_vid)
                                # Triangle area / 3  ; The triangle is 1/2 paralleloid formed by e3, e4
                                voronoi_cell_component = np.linalg.norm(cross_product(e3, e4)) / 6.0
                                voronoi_cell_components.append(voronoi_cell_component)

                    neighbor_weight = -np.sum(cots) / len(cots)  # Normally 1/2(cot a + cot b)
                    v_weights.append(neighbor_weight)  # Weight of wij
                v_weights.append(-np.sum(v_weights))  # Weight of wii

                if self.weighting_scheme == LaplaceBeltramiWeighting.COTANGENT_VORONOI_CELL or \
                        self.weighting_scheme == LaplaceBeltramiWeighting.COTANGENT_ONE_THIRD_TRIANGLE:
                    voronoi_cell_area = sum(voronoi_cell_components)  # 1/Ai (Ai is voronoi cell area)
                    v_weights = [w / voronoi_cell_area for w in v_weights]
                    vertex_areas.append(voronoi_cell_area)
                sparse_mat_weights.append(v_weights)
            self.vertex_areas = vertex_areas

        self.sparse_mat_rows = sparse_mat_rows
        self.sparse_mat_cols = sparse_mat_cols
        self.sparse_mat_weights = sparse_mat_weights

    def add_anchors(self, anchor_idx=None, weight=DEFAULT_ANCHOR_WEIGHT):
        """
        Marks the given vertices as anchors, add as constraint rows to the Laplacian.
        :param anchor_idx: List of integers (vertex ids in the mesh)
        :param weight: Increase to bias the effect of anchor constraints.
        :return: Anchor information is updated in place within this object.
        """
        mesh = self.mesh
        n = mesh.vertices.shape[0]

        if anchor_idx is not None:
            k = len(anchor_idx)
            self.sparse_mat_rows += [[next_anchor_row_idx] for next_anchor_row_idx in range(n, n + k)]
            self.sparse_mat_cols += [[next_anchor_col_idx] for next_anchor_col_idx in anchor_idx]
            if self.weighting_scheme == LaplaceBeltramiWeighting.UMBRELLA or \
                    self.weighting_scheme == LaplaceBeltramiWeighting.COTANGENT_NO_AREA:
                self.sparse_mat_weights += [[weight]] * k
            else:
                self.sparse_mat_weights += [[weight / self.vertex_areas[idx]] for idx in anchor_idx]
            self.anchor_idx.extend(anchor_idx)

    def clear_anchors(self):
        """
        Removes all anchor related information from the Laplacian matrix, effectively leaving it in a newly
        created state.
        :return: This object is updated in place.
        """
        n = self.mesh.vertices.shape[0]
        self.sparse_mat_rows = self.sparse_mat_rows[:n]
        self.sparse_mat_cols = self.sparse_mat_cols[:n]
        self.sparse_mat_weights = self.sparse_mat_weights[:n]
        self.anchor_idx = []

    def get_laplacian(self, normalize):
        """
        Uses the information stored within this object to construct an actual sparse Laplacian matrix.
        :param normalize: True if the rows should be normalized to sum to 1.0 (useful for some applications,
        i.e smoothing & sharpening.
        :return: A sparse (n + k) x n matrix in csr format, where n is the number of vertices in the mesh
                 and k is the number of anchors.
        """
        n = self.mesh.vertices.shape[0]
        k = len(self.anchor_idx)
        sparse_mat_rows = flatten(self.sparse_mat_rows)
        sparse_mat_cols = flatten(self.sparse_mat_cols)

        if normalize:
            sparse_mat_weights = [float(u) / row_weights[-1]
                                  for row_weights in self.sparse_mat_weights
                                  for u in row_weights]
        else:
            sparse_mat_weights = flatten(self.sparse_mat_weights)

        L = sparse.coo_matrix((sparse_mat_weights, (sparse_mat_rows, sparse_mat_cols)), shape=(n + k, n))
        L = L.tocsr()  # Compress the sparse matrix..

        return L
