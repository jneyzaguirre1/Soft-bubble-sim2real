import open3d as o3d
import numpy as np


class MeshGenerator:
    """
    create_box(width=1.0, height=1.0, depth=1.0, create_uv_map=False, map_texture_to_each_face=False)
    create_sphere(radius=1.0, resolution=20, create_uv_map=False)
    create_cylinder(radius=1.0, height=2.0, resolution=20, split=4, create_uv_map=False)
    create_cone(radius=1.0, height=2.0, resolution=20, split=1, create_uv_map=False)
    create_tetrahedron(radius=1.0, create_uv_map=False)
    create_icosahedron(radius=1.0, create_uv_map=False)
    create_octahedron(radius=1.0, create_uv_map=False)
    create_torus(torus_radius=1.0, tube_radius=0.5, radial_resolution=30, tubular_resolution=20)
    create_coordinate_frame(size=1.0, origin=array([0., 0., 0.]))
    create_arrow(cylinder_radius=1.0, cone_radius=1.5, cylinder_height=5.0, cone_height=4.0, resolution=20, cylinder_split=4, cone_split=1)
    """
    def __init__(self):
        pass

    def create_mesh(self, idx):
        if idx == 0: return self.create_box()
        elif idx == 1: return self.create_sphere()
        elif idx == 2: return self.create_cylinder()
        elif idx == 3: return self.create_cone()
        elif idx == 4: return self.create_tetrahedron()
        elif idx == 5: return self.create_icosahedron()
        elif idx == 6: return self.create_octahedron()
        else: return self.create_torus()

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
        height = np.random.uniform(0.2, 1.0)
        params = {'radius':radius, 'height':height}
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=20, split=4, create_uv_map=False)
        return name, params, mesh
    
    def create_cone(self):
        name = 'cone'
        radius = np.random.uniform(0.2, 1.0)
        height = np.random.uniform(0.2, 1.0)
        params = {'radius':radius, 'height':height}
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=20, split=1, create_uv_map=False)
        return name, params, mesh
    
    def create_tetrahedron(self):
        name = 'tetrahedron'
        radius = np.random.uniform(0.2, 1.0)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius=radius, create_uv_map=False)
        return name, params, mesh
    
    def create_icosahedron(self):
        name = 'icosahedron'
        radius = np.random.uniform(0.2, 1.0)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=radius, create_uv_map=False)
        return name, params, mesh
    
    def create_octahedron(self):
        name = 'octahedron'
        radius = np.random.uniform(0.2, 1.0)
        params = {'radius':radius}
        mesh = o3d.geometry.TriangleMesh.create_octahedron(radius=radius, create_uv_map=False)
        return name, params, mesh
    
    def create_torus(self):
        name = 'torus'
        delta = 0.1
        torus_radius = np.random.uniform(0.2, 1.0)
        tube_radius = np.random.uniform(torus_radius/2 - delta/2, torus_radius/2 + delta/2)
        params = {'torus_radius':torus_radius, 'tube_radius':tube_radius}
        mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=torus_radius, tube_radius=tube_radius, radial_resolution=30, tubular_resolution=20)
        return name, params, mesh
    

