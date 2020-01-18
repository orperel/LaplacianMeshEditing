import numpy as np


def joint_faces_graph(mesh):
    graph = mesh.vertex_adjacency_graph

    def _add_edge_info(e, v):
        return [v] if 'facing_vertex' not in graph.edges[e] else graph.edges[e]['facing_vertex'] + [v]

    for (v1, v2, v3) in mesh.faces:
        graph.edges[(v1, v2)]['facing_vertex'] = _add_edge_info(e=(v1, v2), v=v3)
        graph.edges[(v2, v3)]['facing_vertex'] = _add_edge_info(e=(v2, v3), v=v1)
        graph.edges[(v3, v1)]['facing_vertex'] = _add_edge_info(e=(v3, v1), v=v2)
    return graph


def get_vector(mesh, v1_id, v2_id):
    return mesh.vertices[v2_id] - mesh.vertices[v1_id]


def mean2(mesh, v1_id, v2_id):
    return (mesh.vertices[v1_id] + mesh.vertices[v2_id]) / 2


def mean3(mesh, v1_id, v2_id, v3_id):
    return (mesh.vertices[v1_id] + mesh.vertices[v2_id] + mesh.vertices[v3_id]) / 3.0


def cross_product(u, v):
    """ Naive alternative to numpy.cross (too slow in the general case..) """
    return np.array((u[1] * v[2] - u[2] * v[1],
                     u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]))
