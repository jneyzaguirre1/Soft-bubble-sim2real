import os
import numpy as np
from data_collector.data_collector import MyImageDataCollector
    

if __name__ == "__main__":
    kwargs = {}
    my_data_dir = './datasets/toy_dataset/'
    my_data_name = 'toy1'
    img_size = (120, 150)
    kwargs['data_path'] = os.path.join(my_data_dir, my_data_name)
    kwargs['scene_name'] = 'scene_test'
    kwargs['camera_name'] = "camera1"
    kwargs['img_size'] = img_size
    kwargs['cam_int'] = np.array([[250.0, 0, img_size[0]/2],
                                  [0, 250.0, img_size[1]/2],
                                  [0,     0,             1]], dtype=np.double)
    kwargs['cam_ext'] = np.array([[1.0, 0.0, 0.0, 0.0], 
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 6.0],
                                  [0.0, 0.0, 0.0, 1.0]], dtype=np.double)
    kwargs['rot_limits'] = "nothing yet"
    kwargs['trans_limits'] = "nothing yet"
    kwargs['bg_depth'] = 10.0
    kwargs['slope'] = 1.5
    kwargs['obj_count'] = 2
    dc = MyImageDataCollector(**kwargs)
    dc.collect_data(num_data=6)