import numpy as np
from mmint_tools.dataset_tools.data_collection import DataCollectorBase
from mmint_tools.recording_utils.recording_utils import record_image_depth, record_image_color
from data_collector.mesh_generator import MeshGenerator
from render.render_mesh import Renderer


class MyImageDataCollector(DataCollectorBase):

    def __init__(self, *args, **kwargs):
        data_path = kwargs.pop('data_path', None)
        super().__init__(data_path) # TODO: modified

        self.scene_name = kwargs['scene_name']
        self.camera_name = kwargs['camera_name']
        self.img_size = kwargs['img_size']
        self.cam_int = kwargs['cam_int']
        self.cam_ext = kwargs['cam_ext']
        self.rot_limits = kwargs['rot_limits']
        self.trans_limits = kwargs['trans_limits']
        self.bg_depth = kwargs['bg_depth']
        self.slope = kwargs['slope']

        self.obj_idx = 0
        self.obj_count = 0
        self.obj_max_count = kwargs['obj_count']
        self.mesh_generator = MeshGenerator()

        self.render_obj = Renderer(self.img_size, self.cam_int, self.cam_ext, self.bg_depth, self.slope)

    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['SampleIndx', 'SceneName', 'CameraName', 'CameraIntrinsics', 'CameraExtrinsics', 
                        'ImageSize', 'Object', 'ObjectParams', 'Transformation', 'RelativeSlope', 'BackgroundDepth']
        return column_names

    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        legend_line = [data_params['sample_indx'], data_params['scene_name'], data_params['camera_name'], 
                       data_params['camera_intrinsics'], data_params['camera_extrinsics'], data_params['image_size'], 
                       data_params['object_'], data_params['object_params'], data_params['transformation'], 
                       data_params['relative_slope'], data_params['background_depth']]
        legend_lines = [legend_line]
        return legend_lines

    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params: TODO: add what it is!
        Returns: <dict> containing the parameters of the collected sample
        """
        # here, since it is a test, the data will be collected at random
        sample_indx = self.get_new_filecode()
        scene_name = self.scene_name                 
        camera_name = self.camera_name
        camera_intrinsics = self.cam_int
        camera_extrinsics = self.cam_ext
        image_size = self.img_size
        background_depth = self.bg_depth
        object_, object_params, mesh = self.mesh_generator.create_mesh(self.obj_idx)
        H = np.eye(4, dtype=np.double)
        relative_slope = self.slope

        depth_img, transformation = self.render_obj.render_mesh(mesh, H=H)
        depth_gt, depth_sim = self.render_obj.simulate_soft_bubble(depth_img)
        
        depth_gt = depth_gt[..., np.newaxis]        # add dimension so that shape = [120, 150, 1]
        depth_sim = depth_sim[..., np.newaxis]

        # record the image
        record_image_depth(img=depth_gt, save_path=self.data_path, fc=sample_indx, scene_name=scene_name, camera_name=camera_name)
        record_image_depth(img=depth_sim, save_path=self.data_path, fc=sample_indx, scene_name=scene_name, camera_name=camera_name)
        
        sample_params = {
            'sample_indx': sample_indx,
            'scene_name': scene_name,
            'camera_name': camera_name,
            'camera_intrinsics': camera_intrinsics,
            'camera_extrinsics': camera_extrinsics,
            'image_size': image_size,
            'object_': object_,
            'object_params': object_params,
            'transformation': transformation,
            'relative_slope': relative_slope,
            'background_depth': background_depth 
        }

        self.obj_count += 1
        if self.obj_count == self.obj_max_count:
            self.obj_count = 0
            self.obj_idx += 1

        return sample_params