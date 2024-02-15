"""
Code based from answer in stack overflow:
https://stackoverflow.com/questions/67179977/unable-to-render-a-specific-view-of-a-object-in-open3d/67613280#67613280
"""


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import cv2


def soft_bubble_sim(depth_image, bg_depth=10, k_size=11, gk_size=11, sigma=20):
    # Lets start creating the soft edges from the figure
    mask, edges = soft_edges(depth_image, bg_depth, k_size, gk_size, sigma)
    # Now lets combine the edges with the original image, and filter them to smooth the result
    mask_inv = np.float32(mask < 1.0)
    combination = depth_gt * mask_inv + edges
    out = cv2.GaussianBlur(combination, ksize=(3,3), sigmaX=1, sigmaY=1)
    return combination, out

def soft_edges(depth_image, bg_depth=10, k_size=11, gk_size=11, sigma=20):
    mask = np.float32(depth_image < bg_depth)
    kernel = np.ones((k_size, k_size), dtype=np.uint8)
    mask_dil = cv2.dilate(mask, kernel, iterations = 1)
    mask_inv = (1.0 - mask) * mask_dil
    blur = cv2.GaussianBlur(depth_image, ksize=(gk_size,gk_size), sigmaX=sigma, sigmaY=sigma)
    edges = mask_inv * blur
    return mask_inv, edges


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
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # Lets center the figure first
    mesh_center = mesh.get_center()
    mesh.translate(-mesh_center, relative=True)
    
    # Camera properties and position
    img_width, img_height = 120, 150
    intrinsics = np.array([[250.0, 0,  img_width/2],
                           [0, 250.0, img_height/2],
                           [0,     0,            1]], dtype=np.double)
    extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 6.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.double)

    # Create a renderer with the desired image size
    render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)

    # Lets transform the mesh to different locations
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 8, 0, 0*np.pi/4))
    translation = np.array([0, 0, 0], dtype=np.double)
    mesh_t = copy.deepcopy(mesh)
    mesh_t.rotate(R, center=(0, 0, 0))
    mesh_t.translate(translation, relative=True)

    # Define a simple unlit Material.
    mtl = o3d.visualization.rendering.MaterialRecord()  # or Material(), for prior versions of Open3D
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    # Add the mesh to the scene.
    render.scene.add_geometry("rotated_model", mesh_t, mtl)
    render.setup_camera(intrinsics, extrinsics, img_width, img_height)

    # Render to depth from camera position and simulate the soft-bubble output
    bg_depth = 10.0
    depth_gt = np.asarray(render.render_to_depth_image(z_in_view_space=True))
    depth_gt = np.where(np.isfinite(depth_gt), depth_gt, bg_depth)                          # add background
    depth_sim0, depth_sim1 = soft_edges(depth_gt, bg_depth=bg_depth)
    depth_sim2, depth_sim3 = soft_bubble_sim(depth_gt, bg_depth=bg_depth)

    # Display the image in a separate window
    fig, axs = plt.subplots(1, 5, sharex=True, sharey=True)
    axs[0].imshow(depth_gt, cmap='gray_r')
    axs[0].title.set_text('Ground Truth depth image')

    axs[1].imshow(depth_sim0, cmap='gray')
    axs[1].title.set_text('Edges mask')
    
    axs[2].imshow(depth_sim1, cmap='gray')
    axs[2].title.set_text('Edges')
    
    axs[3].imshow(depth_sim2, cmap='gray_r')
    axs[3].title.set_text('Edges + ground truth')
    
    axs[4].imshow(depth_sim3, cmap='gray_r')
    axs[4].title.set_text('Edges + ground truth filtered')
    plt.show()

    # Optionally write it to a PNG file
    #o3d.io.write_image("output.png", depth, 9)
