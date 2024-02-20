import cv2
import numpy as np
import scipy.ndimage


def simulate_soft_bubble(depth_image, bg_depth=10, m=2.5):
    mask_obj = np.where(depth_image != 0, 0, 1).astype(np.float32)
    dist_transform, idx = scipy.ndimage.distance_transform_edt(mask_obj, return_distances=True, return_indices=True)
    dist_transform = dist_transform.astype(np.float32)
    dist_transform_normalized = 1. + m * cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    row_idx, col_idx = idx
    near_edge_value = depth_image[row_idx, col_idx]
    dist_scaled = dist_transform_normalized * near_edge_value
    linear_decay = np.clip(dist_scaled, 0, bg_depth)
    depth_gt = np.where(depth_image != 0, depth_image, bg_depth).astype(np.float32)
    return depth_gt, mask_obj, dist_transform_normalized, dist_scaled, linear_decay

def soft_bubble_sim_old(depth_image, bg_depth=10, k_size=11, sigma=20, c_size=3, c_sigma=1):
    # Lets start creating the soft edges from the figure
    mask_edge, edges = soft_edges(depth_image, bg_depth, k_size, sigma)
    # Now lets combine the edges with the original image, and filter them to smooth the result
    mask_inv = np.float32(mask_edge < 1.0)
    combination = depth_image * mask_inv + edges
    out = cv2.GaussianBlur(combination, ksize=(c_size,c_size), sigmaX=c_sigma, sigmaY=c_sigma)
    return combination, out

def soft_edges(depth_image, bg_depth=10, k_size=11, sigma=20):
    mask_edge = get_edge_mask_dilated(depth_image, bg_depth, k_size)
    blur = cv2.GaussianBlur(depth_image, ksize=(k_size,k_size), sigmaX=sigma, sigmaY=sigma)
    edges = mask_edge * blur
    return mask_edge, edges

def get_edge_mask_dilated(depth_image, bg_depth=10, k_size=11):
    mask_obj = np.float32(depth_image < bg_depth)
    kernel = np.ones((k_size, k_size), dtype=np.uint8)
    mask_dil = cv2.dilate(mask_obj, kernel, iterations = 1)
    mask_edge = (1.0 - mask_obj) * mask_dil
    return mask_edge