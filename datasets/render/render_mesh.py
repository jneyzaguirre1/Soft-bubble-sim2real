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


class Renderer:
    def __init__(self, img_size, intrinsics, extrinsics, bg_depth, slope):
        self.img_size = img_size
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.bg_depth = bg_depth
        self.slope = slope

    def render_mesh(self, mesh, H=np.eye(4, dtype=np.double)):
        # Lets center the figure first
        mesh_center = mesh.get_center()
        mesh.translate(-mesh_center, relative=True)
        
        # Image size
        img_width, img_height = self.img_size

        # Create a renderer with the desired image size
        render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
        scene = render.scene

        # Lets transform the mesh to a different location
        mesh_t, H_co = self.transform_mesh(mesh, H)

        # Define a simple unlit Material.
        mtl = o3d.visualization.rendering.MaterialRecord()  # or Material(), for prior versions of Open3D
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"

        # Add the mesh to the scene.
        render.scene.add_geometry("transformed_model", mesh_t, mtl)
        render.setup_camera(self.intrinsics, self.extrinsics, self.img_size[1], self.img_size[0])

        # Render to depth from camera position and simulate the soft-bubble output
        depth_no_bg = np.asarray(render.render_to_depth_image(z_in_view_space=True))
        depth_no_bg = np.where(np.isfinite(depth_no_bg), depth_no_bg, 0)                   # add background
        return depth_no_bg, H_co

    def transform_mesh(self, mesh, H):
        mesh_t = copy.deepcopy(mesh)
        mesh_t.transform(H)
        H_co = self.extrinsics @ H   # RBT between from camera to obj.
        return mesh_t, H_co
    
    def simulate_soft_bubble(self, depth_image, DEBUG=False):
        mask_obj = np.where(depth_image != 0, 0, 1).astype(np.float32)
        dist_transform, idx = scipy.ndimage.distance_transform_edt(mask_obj, return_distances=True, return_indices=True)
        dist_transform = dist_transform.astype(np.float32)
        dist_transform_normalized = 1. + self.slope * cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
        row_idx, col_idx = idx
        near_edge_value = depth_image[row_idx, col_idx]
        dist_scaled = dist_transform_normalized * near_edge_value
        linear_decay = np.clip(dist_scaled, 0, self.bg_depth)
        depth_gt = np.where(depth_image != 0, depth_image, self.bg_depth).astype(np.float32)
        if DEBUG:
            return depth_gt, mask_obj, dist_transform_normalized, dist_scaled, linear_decay
        else:
            return depth_gt, linear_decay


if __name__ == "__main__":
    # params:
    BG_DEPTH = 10.0     # background depth
    m = 2.5             # Relative slope

    img_size = (120, 150)
    intrinsics = np.array([[250.0, 0, img_size[0]/2],
                           [0, 250.0, img_size[1]/2],
                           [0,     0,             1]], dtype=np.double)
    extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 6.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.double)
    
    render_obj = Renderer(img_size, intrinsics, extrinsics, BG_DEPTH, m)

    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    
    
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi/4))
    t = np.array([0, 0, 0], dtype=np.double).reshape(3,1)
    H = np.block([[R, t], [np.zeros((1,3), dtype=np.double), 1.0]])

    depth_no_bg, H_co = render_obj.render_mesh(mesh, H=H)

    depth_gt, depth_sim0, depth_sim1, depth_sim2, depth_sim3 = render_obj.simulate_soft_bubble(depth_no_bg, DEBUG=True)

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
