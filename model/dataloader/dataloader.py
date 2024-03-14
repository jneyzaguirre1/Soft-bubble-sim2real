import os
import torch
from torch.utils.data import Dataset, DataLoader


class BubbleDataset(Dataset):

    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the .pt files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # List all the .pt files in the directory
        self.file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
        self.transform = transform

    def __len__(self):
        # Return the number of .pt files
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the .pt file at the specified index
        sample = torch.load(self.file_paths[idx])
        
        fc_i = sample['sample_indx']
        depth_gt = sample['depth_gt']
        depth_img = sample['depth_img']
        intrinsics = sample['camera_intrinsics']
        extrinsics = sample['camera_extrinsics']
        img_size = sample['image_size']
        object_name = sample['object_name']
        object_params = sample['object_params']
        transformation = sample['transformation']
        slope = sample['relative_slope']
        bg_depth = sample['background_depth']
        
        # Apply the transform if specified
        if self.transform:
            depth_gt = self.transform(depth_gt)
            depth_img = self.transform(depth_img)

        return depth_gt, depth_img, intrinsics, extrinsics, img_size, object_name, object_params, transformation, slope, bg_depth