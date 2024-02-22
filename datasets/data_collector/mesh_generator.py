import open3d as o3d
import numpy as np


class MeshGenerator:
    """
    create_arrow(cylinder_radius=1.0, cone_radius=1.5, cylinder_height=5.0, cone_height=4.0, resolution=20, cylinder_split=4, cone_split=1)
    create_box(width=1.0, height=1.0, depth=1.0, create_uv_map=False, map_texture_to_each_face=False)
    create_cone(radius=1.0, height=2.0, resolution=20, split=1, create_uv_map=False)
    create_coordinate_frame(size=1.0, origin=array([0., 0., 0.]))
    create_cylinder(radius=1.0, height=2.0, resolution=20, split=4, create_uv_map=False)
    create_icosahedron(radius=1.0, create_uv_map=False)
    create_octahedron(radius=1.0, create_uv_map=False)
    create_sphere(radius=1.0, resolution=20, create_uv_map=False)
    create_torus(torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20)
    create_tetrahedron(radius=1.0, create_uv_map=False)
    """
    def __init__(self):
        pass

    def create_mesh(self, idx):
        if idx == 0: return self.create_box()
        elif idx ==1: return self.create_sphere()
        else: return self.create_cylinder()

    def create_box(self):
        name = 'box'
        box_size = np.random.uniform(0.2, 1.0, size=3)
        params = {'box_size':box_size}
        mesh = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2], create_uv_map=False, map_texture_to_each_face=False)
        return name, params, mesh

    def create_sphere(self):
        name = 'sphere'
        radius = np.random.uniform(0.2, 1.0)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20, create_uv_map=False)
        return name, params, mesh

    def create_cylinder(self):
        name = 'cylinder'
        radius = np.random.uniform(0.2, 1.0)
        height = np.random.uniform(0.2, 2.0)
        params = {'radius':radius, 'height':height}
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=20, split=4, create_uv_map=False)
        return name, params, mesh
