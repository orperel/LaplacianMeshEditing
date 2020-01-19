import copy
import numpy as np
from scipy import sparse
from lib.laplacian_beltrami import LaplacianBeltrami, LaplaceBeltramiWeighting, DEFAULT_ANCHOR_WEIGHT


def mean_curvature(mesh, weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA):
    """ Calculates mean curvature scalar field over the mesh, per vertex """
    LB_operator = LaplacianBeltrami(mesh, weighting_scheme)
    L = LB_operator.get_laplacian(normalize=False)
    v = mesh.vertices
    mean_curvature = np.linalg.norm(L * v, axis=1)
    return mean_curvature


def laplacian_smoothing(mesh, smooth_factor=1.0, iterations=1,
                        weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA):
    for _ in range(iterations):
        LB_operator = LaplacianBeltrami(mesh, weighting_scheme)  # Weighting may change between iterations..
        L = LB_operator.get_laplacian(normalize=True)
        v = np.array(mesh.vertices)
        diff = L * v
        mesh.vertices = v - smooth_factor * diff


def laplacian_sharpening(mesh, sharpen_factor=1.0, iterations=1,
                         weighting_scheme=LaplaceBeltramiWeighting.UMBRELLA):
    for _ in range(iterations):
        LB_operator = LaplacianBeltrami(mesh, weighting_scheme)  # Weighting may change between iterations..
        L = LB_operator.get_laplacian(normalize=True)
        v = np.array(mesh.vertices)
        diff = L * v
        mesh.vertices = v + sharpen_factor * diff


def _solve_lsqr(mesh, L, anchors, anchor_idx):
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
    LB_operator = LaplacianBeltrami(mesh, weighting_scheme)
    LB_operator.add_anchors(anchor_idx, anchors_weight)
    L = LB_operator.get_laplacian(normalize=False)

    _solve_lsqr(mesh, L, anchors, anchor_idx)
    return LB_operator


def resolve_laplace_matrix(mesh, LB_operator, anchors=None, anchor_idx=None, anchors_weight=DEFAULT_ANCHOR_WEIGHT):
    LB_operator.clear_anchors()
    LB_operator.add_anchors(anchor_idx, anchors_weight)
    L = LB_operator.get_laplacian(normalize=False)

    _solve_lsqr(mesh, L, anchors, anchor_idx)
    return LB_operator


def spectral_decomposition(mesh, weighting_scheme, K, eps=1e-14):
    LB_operator = LaplacianBeltrami(mesh, weighting_scheme)
    L = LB_operator.get_laplacian(normalize=False)

    robust_L = L * (sparse.eye(*L.shape) * eps)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(robust_L, K, which='LM', sigma=0)

    return eigenvalues, eigenvectors


def mesh_reconstruct(mesh, eigvectors, vec_count_to_use):
    mesh = mesh.copy()  # Maintain the original
    v = mesh.vertices
    reconstruct_vecs = eigvectors[:, :vec_count_to_use]
    updated_mesh = reconstruct_vecs @ reconstruct_vecs.transpose() @ v
    mesh.vertices = updated_mesh
    return mesh