import os
import numpy as np
from data_collector.data_collector import MyImageDataCollector
    

if __name__ == "__main__":
    OBJS = 8
    IMAGE_PAIRS_PER_OBJ = 1250
    my_data_dir = './datasets/toy_dataset/'
    my_data_name = 'serious_toyX'
    img_size = (120, 150)

    kwargs = {}
    kwargs['obj_index'] = None          # specify just one type of object!
    kwargs['data_path'] = os.path.join(my_data_dir, my_data_name)
    kwargs['scene_name'] = 'scene_test'
    kwargs['camera_name'] = "depth_gt"
    kwargs['bubble_camera_name'] = "depth_soft_bubble"
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
    kwargs['debug'] = False
    kwargs['obj_count'] = IMAGE_PAIRS_PER_OBJ
    
    dc = MyImageDataCollector(**kwargs)
    num_data = OBJS * IMAGE_PAIRS_PER_OBJ
    dc.collect_data(num_data=num_data)