import numpy as np
from mmint_tools.dataset_tools.legended_dataset import LegendedDataset
from mmint_tools.data_utils.loading_utils import load_image_color, load_image_depth, load_array


class MyImgDataset(LegendedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _get_sample_codes(self):
        """
        Return a list containing the data filecodes.
        Overwrite the function in case the data needs to be filtered.
        By default we load all the filecodes we find in the datalegend
        :return:
        """
        return np.arange(len(self.dl))

    def _get_sample(self, sample_code):
        """
        Returns the sample corresponding to the filecode fc
        :param sample_code:
        :return:
        """
        dl_line = self.dl.iloc[sample_code]
        fc_i = dl_line['SampleIndx']
        scene_name = dl_line['SceneName']
        camera_name = dl_line['CameraName']
        bubble_cam_name = dl_line['BubbleCameraName']
        img_size = dl_line['ImageSize']
        object_name = dl_line['ObjectName']
        object_params = dl_line['ObjectParams']
        slope = dl_line['RelativeSlope']
        bg_depth = dl_line['BackgroundDepth']
                
        depth_gt = load_image_depth(self.data_path, fc=fc_i, scene_name=scene_name, camera_name=camera_name)
        depth_img = load_image_depth(self.data_path, fc=fc_i,  scene_name=scene_name, camera_name=bubble_cam_name)
        
        intrinsics = load_array(self.data_path, fc=fc_i, scene_name=scene_name, array_name="camera_intrinsics")
        extrinsics = load_array(self.data_path, fc=fc_i, scene_name=scene_name, array_name="camera_extrinsics")
        transformation = load_array(self.data_path, fc=fc_i, scene_name=scene_name, array_name="transformation")

        sample_i = {
            'sample_indx': fc_i,
            'depth_gt': depth_gt,
            'depth_img': depth_img,
            'camera_intrinsics': intrinsics,
            'camera_extrinsics': extrinsics,
            'image_size': img_size,
            'object_name': object_name,
            'object_params': object_params,
            'transformation': transformation,
            'relative_slope': float(slope),
            'background_depth': float(bg_depth) 
        }
        return sample_i

    def get_name(self):
        return self.data_name