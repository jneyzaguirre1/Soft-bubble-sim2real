import numpy as np
from mmint_tools.dataset_tools.data_collection import DataCollectorBase
from mmint_tools.recording_utils.recording_utils import record_image_depth, record_image_color, record_array
from data.data_collector.mesh_generator import MeshGenerator
from data.render.render_mesh import Renderer
import matplotlib.pyplot as plt


class MyImageDataCollector(DataCollectorBase):

    def __init__(self, *args, **kwargs):
        data_path = kwargs.pop('data_path', None)
        super().__init__(data_path) # TODO: modified

        self.scene_name = kwargs['scene_name']
        self.cam_name = kwargs['camera_name']                   # beauty depth
        self.bubble_cam_name = kwargs['bubble_camera_name']     # soft bubble depth
        self.img_size = kwargs['img_size']
        self.cam_int = kwargs['cam_int']
        self.cam_ext = kwargs['cam_ext']
        self.rot_limits = kwargs['rot_limits']
        self.trans_limits = kwargs['trans_limits']
        self.bg_depth = kwargs['bg_depth']
        self.slope = kwargs['slope']
        self.DEBUG = kwargs['debug']
        self.obj_index = kwargs['obj_index']

        self.obj_idx = 0
        self.obj_count = 0
        self.obj_max_count = kwargs['obj_count']
        self.mesh_generator = MeshGenerator()

        self.render_obj = Renderer(self.img_size, self.cam_int, self.cam_ext, self.bg_depth, self.slope, debug=self.DEBUG)

    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['SampleIndx', 'SceneName', 'CameraName', "BubbleCameraName",'CameraIntrinsics', 'CameraExtrinsics', 
                        'ImageSize', 'ObjectName', 'ObjectParams', 'Transformation', 'RelativeSlope', 'BackgroundDepth']
        return column_names

    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        legend_line = [data_params['sample_indx'], data_params['scene_name'], data_params['camera_name'], data_params['bubble_camera_name'],
                       data_params['camera_intrinsics'], data_params['camera_extrinsics'], data_params['image_size'], 
                       data_params['object_name'], data_params['object_params'], data_params['transformation'], 
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
        #breakpoint()
        sample_indx = self.get_new_filecode()
        camera_intrinsics = self.cam_int
        camera_extrinsics = self.cam_ext
        if self.obj_index is not None: self.obj_idx = self.obj_index

        object_name, object_params, mesh = self.mesh_generator.create_mesh(self.obj_idx)
        ret = False
        while not ret:                  # runs until a valid image is rendered
            H = self.render_obj.get_transformation(mesh)
            ret1, depth_img, transformation = self.render_obj.render_mesh(mesh, H=H)
            ret2, depth_gt, depth_sim = self.render_obj.simulate_soft_bubble(depth_img)
            ret = ret1 and ret2
            if self.DEBUG:
                text = "".join([f"{key} : {value}, " for key, value in object_params.items()])
                fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                fig.suptitle(f"{object_name} {self.obj_count}: {text}, t : {H[0:3,3]}")
                axs[0].imshow(depth_gt, cmap='gray_r')
                axs[0].title.set_text("Ground Truth depth")
                axs[1].imshow(depth_sim, cmap='gray_r')
                axs[1].title.set_text(f"Output, ret: {ret}")
                plt.show()

        depth_gt = depth_gt[..., np.newaxis]        # add dimension so that shape = [120, 150, 1]
        depth_sim = depth_sim[..., np.newaxis]

        # record the image
        record_image_depth(img=depth_gt, save_path=self.data_path, fc=sample_indx, scene_name=self.scene_name, camera_name=self.cam_name)
        record_image_depth(img=depth_sim, save_path=self.data_path, fc=sample_indx, scene_name=self.scene_name, camera_name=self.bubble_cam_name)
        
        record_array(camera_extrinsics, save_path=self.data_path, fc=sample_indx, scene_name=self.scene_name, array_name=f"{camera_extrinsics=}".split("=")[0])
        record_array(camera_intrinsics, save_path=self.data_path, fc=sample_indx, scene_name=self.scene_name, array_name=f"{camera_intrinsics=}".split("=")[0])
        record_array(transformation, save_path=self.data_path, fc=sample_indx, scene_name=self.scene_name, array_name=f"{transformation=}".split("=")[0])

        sample_params = {
            'sample_indx': sample_indx,
            'scene_name': self.scene_name,
            'camera_name': self.cam_name,
            'bubble_camera_name' : self.bubble_cam_name,
            'camera_intrinsics': camera_intrinsics,
            'camera_extrinsics': camera_extrinsics,
            'image_size': self.img_size,
            'object_name': object_name,
            'object_params': object_params,
            'transformation': transformation,
            'relative_slope': self.slope,
            'background_depth': self.bg_depth 
        }

        self.obj_count += 1
        if self.obj_count == self.obj_max_count:
            self.obj_count = 0
            self.obj_idx += 1

        return sample_params