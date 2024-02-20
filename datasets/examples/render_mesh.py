"""
Code based from answer in stack overflow:
https://stackoverflow.com/questions/67179977/unable-to-render-a-specific-view-of-a-object-in-open3d/67613280#67613280
"""


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import cv2
import scipy.ndimage
from math import ceil
from soft_bubble_sim import *


def render_mesh(mesh, img_size, intrinsics, extrinsics, H=np.eye(4, dtype=np.double)):
    # Lets center the figure first
    mesh_center = mesh.get_center()
    mesh.translate(-mesh_center, relative=True)
    
    # Image size
    img_width, img_height = img_size

    # Create a renderer with the desired image size
    render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
    scene = render.scene

    # Lets transform the mesh to a different location
    mesh_t = copy.deepcopy(mesh)
    mesh_t.transform(H)
    H_co = extrinsics @ H   # RBT between from camera to obj.

    # Define a simple unlit Material.
    mtl = o3d.visualization.rendering.MaterialRecord()  # or Material(), for prior versions of Open3D
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # Add the mesh to the scene.
    render.scene.add_geometry("transformed_model", mesh_t, mtl)
    render.setup_camera(intrinsics, extrinsics, img_width, img_height)

    # Render to depth from camera position and simulate the soft-bubble output
    depth_no_bg = np.asarray(render.render_to_depth_image(z_in_view_space=True))
    depth_no_bg = np.where(np.isfinite(depth_no_bg), depth_no_bg, 0)                   # add background
    return depth_no_bg, H_co


if __name__ == "__main__":
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
    # params:
    BG_DEPTH = 10.0     # background depth
    m = 2.5             # Relative slope

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    img_size = (120, 150)
    intrinsics = np.array([[250.0, 0, img_size[0]/2],
                           [0, 250.0, img_size[1]/2],
                           [0,     0,             1]], dtype=np.double)
    extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 6.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.double)
    
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi/4))
    t = np.array([0, 0, 0], dtype=np.double).reshape(3,1)
    H = np.block([[R, t], [np.zeros((1,3), dtype=np.double), 1.0]])

    depth_no_bg, H_co = render_mesh(mesh, img_size, intrinsics, extrinsics, H=H)

    depth_gt, depth_sim0, depth_sim1, depth_sim2, depth_sim3 = simulate_soft_bubble(depth_no_bg, bg_depth=BG_DEPTH, m=m)

    # Display the image in a separate window
    fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
    fig.suptitle(f"Edge linear decay process for a relative slope of {m}")
    axs[0].imshow(depth_gt, cmap='gray_r')
    axs[0].title.set_text('Ground Truth depth')

    axs[1].imshow(depth_sim0, cmap='gray_r')
    axs[1].title.set_text('Object mask')
    
    axs[2].imshow(depth_sim1, cmap='gray_r')
    axs[2].title.set_text('Norm dist. transf.')
    
    axs[3].imshow(depth_sim2, cmap='gray_r')
    axs[3].title.set_text('Scale dist. transf.')
    
    axs[4].imshow(depth_sim3, cmap='gray_r')
    axs[4].title.set_text('Scale dist. transf. clipped to bg')
    plt.show()
