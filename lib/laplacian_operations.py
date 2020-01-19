import numpy as np
from scipy import sparse
from lib.laplacian_beltrami import LaplacianBeltrami, LaplaceBeltramiWeighting, DEFAULT_ANCHOR_WEIGHT


def mean_curvature(mesh, weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA):
    """
    Calculates mean curvature scalar field over the mesh, per vertex.
    Mean curvature is defined as |L * v|,
    where L is the n x n Laplace operator of the mesh and v is a n x 3 vector of mesh vertex positions.
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param weighting_scheme: Any of LaplaceBeltramiWeighting, determines how geometry is induced on LB.
    :return: A scalar field of n entries, representing the mean curvature at each vertex.
    """
    LB_operator = LaplacianBeltrami(mesh, weighting_scheme)
    L = LB_operator.get_laplacian(normalize=False)
    v = mesh.vertices
    mean_curvature = np.linalg.norm(L * v, axis=1)
    return mean_curvature


def laplacian_smoothing(mesh, smooth_factor=1.0, iterations=1,
                        weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA):
    """
    Performs smoothing over the mesh using the Laplace-Beltrami operator.
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param smooth_factor: Strength factor of how much smoothing to apply.
                          Normally should be around 0.0 ~ 2.0.
    :param iterations: Number of iterations the process is repeated.
    :param weighting_scheme: Any of LaplaceBeltramiWeighting, determines how geometry is induced on LB.
    :return: The mesh is updated in place.
    """
    for _ in range(iterations):
        LB_operator = LaplacianBeltrami(mesh, weighting_scheme)  # Weighting may change between iterations..
        L = LB_operator.get_laplacian(normalize=True)
        v = np.array(mesh.vertices)
        diff = L * v
        mesh.vertices = v - smooth_factor * diff


def laplacian_sharpening(mesh, sharpen_factor=1.0, iterations=1,
                         weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA):
    """
    Performs sharpening over the mesh using the Laplace-Beltrami operator.
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param smooth_factor: Strength factor of how much sharpening to apply.
                          Normally should be around 0.0 ~ 2.0.
    :param iterations: Number of iterations the process is repeated.
    :param weighting_scheme: Any of LaplaceBeltramiWeighting, determines how geometry is induced on LB.
    :return: The mesh is updated in place.
    """
    for _ in range(iterations):
        LB_operator = LaplacianBeltrami(mesh, weighting_scheme)  # Weighting may change between iterations..
        L = LB_operator.get_laplacian(normalize=True)
        v = np.array(mesh.vertices)
        diff = L * v
        mesh.vertices = v + sharpen_factor * diff


def _solve_lsqr(mesh, L, anchors, anchor_idx):
    """
    Solves the over-constrained equations system using least squares, defined by the following equations:
    L * v = delta
    anchor_weight * v_anchors = anchor_weight * anchors
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param L: Laplacian matrix, obtained from an instance of the LaplacianBeltrami class.
    :param anchors: List of np.arrays of size 3, representing positions of anchors in space, introduced as
    constraints (and boundary conditions) to the equations system.
    :param anchor_idx: Indices of the anchors as vertex ids within the mesh.
    :return: The equations system is solved and the mesh's positions are updated in place.
    """
    v = np.array(mesh.vertices)
    delta = L * v

    n = v.shape[0]
    k = len(anchor_idx)
    for i in range(k):
        weight = np.sum(L[n + i])
        delta[n + i, :] = weight * anchors[i]

    # Solve each axis separately
    for i in range(3):
        mesh.vertices[:, i] = sparse.linalg.lsqr(L, delta[:, i])[0]


def solve_laplace_matrix(mesh, anchors=None, anchor_idx=None,
                         weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA,
                         anchors_weight=DEFAULT_ANCHOR_WEIGHT):
    """
    Solves the over-constrained equations system using least squares, defined by the following equations:
    L * v = delta
    anchor_weight * v_anchors = anchor_weight * anchors
    The system is solved to obtain new positions that satisfy this optimization problem.
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param anchors: List of np.arrays of size 3, representing positions of anchors in space, introduced as
    :param anchor_idx: Indices of the anchors as vertex ids within the mesh.
    :param weighting_scheme: Any of LaplaceBeltramiWeighting, determines how geometry is induced on LB.
    :param anchors_weight: Increase to bias the optimization problem towards the anchor constraints.
    :return: Laplace Beltrami operator created in the process (for further reuse)
    """
    LB_operator = LaplacianBeltrami(mesh, weighting_scheme)
    LB_operator.add_anchors(anchor_idx, anchors_weight)
    L = LB_operator.get_laplacian(normalize=False)

    _solve_lsqr(mesh, L, anchors, anchor_idx)
    return LB_operator


def resolve_laplace_matrix(mesh, LB_operator, anchors=None, anchor_idx=None, anchors_weight=DEFAULT_ANCHOR_WEIGHT):
    """
    Solves the over-constrained equations system using least squares, defined by the following equations:
    L * v = delta
    anchor_weight * v_anchors = anchor_weight * anchors
    The system is solved to obtain new positions that satisfy this optimization problem.
    Use this formulation if the same LaplaceBeltrami instance is used to solve lsqr more than once.
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param anchors: List of np.arrays of size 3, representing positions of anchors in space, introduced as
    :param anchor_idx: Indices of the anchors as vertex ids within the mesh.
    :param weighting_scheme: Any of LaplaceBeltramiWeighting, determines how geometry is induced on LB.
    :param anchors_weight: Increase to bias the optimization problem towards the anchor constraints.
    :return: Laplace Beltrami operator used in the process (for further reuse)
    """
    LB_operator.clear_anchors()
    LB_operator.add_anchors(anchor_idx, anchors_weight)
    L = LB_operator.get_laplacian(normalize=False)

    _solve_lsqr(mesh, L, anchors, anchor_idx)
    return LB_operator


def spectral_decomposition(mesh, weighting_scheme, K, eps=1e-14):
    """
    Performs Spectral Decomposition over the mesh Laplacian.
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param weighting_scheme: Any of LaplaceBeltramiWeighting, determines how geometry is induced on LB.
    :param K: First K eigenvectors to return.
    :param eps: Epsilon value used to ensure the decomposed matrix is not singular.
    :return: A tuple of eigenvalues, eigenvectors, where:
            eigenvalues is a numpy array of n floats
            eigenvectors is a numpy array of n x K
            Both are sorted in ascending order of eigenvalues magnitude
    """
    LB_operator = LaplacianBeltrami(mesh, weighting_scheme)
    L = LB_operator.get_laplacian(normalize=False)

    robust_L = L * (sparse.eye(*L.shape) * eps)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(robust_L, K, which='LM', sigma=0)

    return eigenvalues, eigenvectors


def mesh_reconstruct(mesh, eigvectors, vec_count_to_use):
    """
    Performs mesh reconstruction using M first eigenvectors of the mesh Laplacian.
    :param mesh: 3d triangular mesh, in Trimesh format.
    :param eigvectors: Numpy array of eigenvalues, n x K
    :param vec_count_to_use: M, specifying how many M < K eigenvectors should be used to reconstruct the mesh.
    :return: A copy of the mesh, reconstructed from M eigenvectors.
    """
    mesh = mesh.copy()  # Maintain the original
    v = mesh.vertices
    reconstruct_vecs = eigvectors[:, :vec_count_to_use]
    updated_mesh = reconstruct_vecs @ reconstruct_vecs.transpose() @ v
    mesh.vertices = updated_mesh
    return mesh
