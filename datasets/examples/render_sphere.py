"""
Code based in the example provided by Open3d at:
https://www.open3d.org/docs/latest/python_example/geometry/point_cloud/index.html#point-cloud-to-depth-py
"""

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
    pcd = mesh.sample_points_uniformly(number_of_points=100000)

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
                                 [0, 0, 1]])
    extrinsics = o3d.core.Tensor([[1.0, 0.0, 0.0, 0.0], 
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 6.0],
                                  [0.0, 0.0, 0.0, 1.0]])
    print(type(pcd))
    points_tensor = o3d.core.Tensor(np.asarray(pcd.points, dtype=np.float32))
    pcd_tensor = o3d.t.geometry.PointCloud(points_tensor)
    o3d.visualization.draw([pcd_tensor])
    depth_reproj = pcd_tensor.project_to_depth_image(640,
                                              480,
                                              intrinsic,
                                              extrinsics,
                                              depth_scale=1.0,
                                              depth_max=5.0)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.asarray(depth_reproj.to_legacy()))
    axs[1].imshow(np.asarray(depth_reproj.to_legacy()))
    plt.show()