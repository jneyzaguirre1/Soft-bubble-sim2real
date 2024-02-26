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

    def __init__(self, img_size, intrinsics, extrinsics, bg_depth, slope, debug=False):
        self.img_size = img_size
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.bg_depth = bg_depth
        self.slope = slope
        self.DEBUG = debug
        self.MIN_PIXEL_COUNT = 100

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
        render.setup_camera(self.intrinsics, self.extrinsics, img_width, img_height)

        # Render to depth from camera position and simulate the soft-bubble output
        depth_no_bg = np.asarray(render.render_to_depth_image(z_in_view_space=True))
        depth_no_bg = np.where(np.isfinite(depth_no_bg), depth_no_bg, 0)                   # add background
        ret = self.check_pixel_data_in_render(depth_no_bg)
        return ret, depth_no_bg, H_co

    def transform_mesh(self, mesh, H):
        mesh_t = copy.deepcopy(mesh)
        mesh_t.transform(H)
        H_co = self.extrinsics @ H   # RBT between from camera to obj.
        return mesh_t, H_co
    
    def simulate_soft_bubble(self, depth_image, DEBUG=False):
        mask_obj = np.where(depth_image != 0, 0, 1).astype(np.float32)
        dist_transform, idx = scipy.ndimage.distance_transform_edt(mask_obj, return_distances=True, return_indices=True)
        dist_transform_normed = cv2.normalize(dist_transform.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        dist_transform_sloped = 1. + self.slope * dist_transform_normed
        row_idx, col_idx = idx
        near_edge_value = depth_image[row_idx, col_idx]
        dist_scaled = dist_transform_sloped * near_edge_value
        out = np.clip(dist_scaled, 0, self.bg_depth)
        depth_gt = np.where(depth_image != 0, depth_image, self.bg_depth).astype(np.float32)
        ret = self.check_obj_in_render(out, self.bg_depth)
        if DEBUG:
            return ret, depth_gt, mask_obj, dist_transform_normed, dist_transform_sloped, dist_scaled, out
        else:
            return ret, depth_gt, out
        
    def check_obj_in_render(self, depth_image, comp_value):
        """checks if there is only background in the depth image"""
        ret = not np.all(depth_image == comp_value)
        if self.DEBUG and not ret: print("[W] Object not in render")
        return ret
    
    def check_pixel_data_in_render(self, depth_image):
        """checks if there is enought information in the image"""
        count = np.count_nonzero(depth_image)
        ret = count >= self.MIN_PIXEL_COUNT
        if self.DEBUG:
            if not ret: print("[W] Not enought pixel information in depth image")
            print(f"Pixel Count: {count}")
        return ret
    
    def get_transformation(self, mesh):
        t_low = np.array([-1.5, -1.85, -2], dtype=np.double)
        t_high = np.array([1.5, 1.85, 3.5], dtype=np.double)
        t = np.random.uniform(low=t_low, high=t_high, size=3).reshape(3,1)
        R_low = -1 * np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        R_high = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        R_ = np.random.uniform(low=R_low, high=R_high, size=3).reshape(3,1)
        R = mesh.get_rotation_matrix_from_xyz(tuple(R_))
        H = np.block([[R, t], [np.zeros((1,3), dtype=np.double), 1.0]])
        return H


if __name__ == "__main__":
    # params:
    BG_DEPTH = 10.0     # background depth
    m = 1.5             # Relative slope
    m2 = 2.5

    img_size = (120, 150)
    intrinsics = np.array([[250.0, 0, img_size[0]/2],
                           [0, 250.0, img_size[1]/2],
                           [0,     0,             1]], dtype=np.double)
    extrinsics = np.array([[1.0, 0.0, 0.0, 0.0], 
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 6.0],
                           [0.0, 0.0, 0.0, 1.0]], dtype=np.double)
    
    render_obj = Renderer(img_size, intrinsics, extrinsics, BG_DEPTH, m)
    render_obj2 = Renderer(img_size, intrinsics, extrinsics, BG_DEPTH, m2)
    
    a = 1.0
    mesh = o3d.geometry.TriangleMesh.create_box(0.88, 0.2, 0.89)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    R = mesh.get_rotation_matrix_from_xyz((0 * np.pi / 4, 0 * np.pi / 2, 0 * np.pi / 4))
    #t = np.array([-1.43, -1.7, -1.65], dtype=np.double).reshape(3,1)
    t = np.array([0, 0, 0], dtype=np.double).reshape(3,1)
    H = np.block([[R, t], [np.zeros((1,3), dtype=np.double), 1.0]])

    ret1, depth_no_bg, H_co = render_obj.render_mesh(mesh, H=H)

    _, depth_no_bg2, H_co2 = render_obj2.render_mesh(mesh, H=H)

    ret2, depth_gt, depth_sim0, depth_sim1, depth_sim2, depth_sim3, depth_sim4 = render_obj.simulate_soft_bubble(depth_no_bg, DEBUG=True)
    _, _, depth_sim4_2 = render_obj2.simulate_soft_bubble(depth_no_bg)
    ret = ret1 and ret2
    print(ret)
    # Display the image in a separate window
    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # #fig.suptitle(f"Edge linear decay process for a relative slope of {m}")
    # axs[0].imshow(depth_gt, cmap='gray_r')
    # axs[0].title.set_text('Ground Truth depth')

    # axs[1].imshow(depth_sim0, cmap='gray_r')
    # axs[1].title.set_text('Object mask')
    
    # fig1, axs1 = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs1[0].imshow(depth_sim1, cmap='gray_r')
    # axs1[0].title.set_text('Normed distance transf.')
    
    # axs1[1].imshow(depth_sim2, cmap='gray_r')
    # axs1[1].title.set_text('Slope adjusted distance transf.')
    
    # fig2, axs2 = plt.subplots(1, 2, sharex=True, sharey=True)
    # axs2[0].imshow(depth_sim3, cmap='gray_r')
    # axs2[0].title.set_text('Edge adjusted decay')
    
    # axs2[1].imshow(depth_sim4, cmap='gray_r')
    # axs2[1].title.set_text('Background clipped output')

    fig3, axs3 = plt.subplots(1, 3, sharex=True, sharey=True)
    axs3[0].imshow(depth_gt, cmap='gray_r')
    axs3[0].title.set_text('Ground Truth depth')
    
    axs3[1].imshow(depth_sim3, cmap='gray_r')
    axs3[1].title.set_text(f"Output, slope={m}")

    axs3[2].imshow(depth_sim4, cmap='gray_r')
    axs3[2].title.set_text(f"Output, slope={m2}")
    
    # plt.figure()
    # plt.imshow(depth_gt, cmap='gray_r')
    # plt.title('Ground Truth Depth')
    
    plt.show()
