from vedo import *


def load_triangular_mesh(mesh_path,
                         is_preprocess=True,
                         keep_only_largest_component=False,
                         only_watertight=False):
    """
    Loads the mesh using vtkplotter, triangulates it and returns as a trimesh
    :param mesh_path: String representing location of mesh object to load
    :param is_preprocess: If True, Trimesh will perform some preprocessing to make sure the mesh is valid
           (e.g: no degenerated triangles, no nans or exploding vals, etc).
    :param keep_only_largest_component: Discard all smaller components within the mesh, and work only on largest one.
    :return: Trimesh of loaded object, with triangulated faces
    """
    mesh = load(mesh_path)
    mesh = mesh.triangulate()  # Ensure the mesh is triangular..
    triangular_mesh = vedo2trimesh(mesh)
    if keep_only_largest_component:
        triangular_mesh = triangular_mesh.split(only_watertight=only_watertight)[0]
    if is_preprocess:
        triangular_mesh = triangular_mesh.process()
        triangular_mesh.remove_degenerate_faces()
    return triangular_mesh


def present_mesh(triangular_mesh):
    """
    Converts Trimesh to vtkplotter format and presents it
    :param triangular_mesh: Trimesh of model
    """
    output_mesh = trimesh2vedo(triangular_mesh)
    show(output_mesh)
    interactive()
