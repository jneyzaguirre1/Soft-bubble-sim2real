import os
import torch
from torch.utils.data import Dataset


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
        """
        sample['sample_indx']
        sample['depth_gt']
        sample['depth_img']
        sample['camera_intrinsics']
        sample['camera_extrinsics']
        sample['image_size']
        sample['object_name']
        sample['object_params']
        sample['transformation']
        sample['relative_slope']
        sample['background_depth']
        """
        # Load the .pt file at the specified index
        sample = torch.load(self.file_paths[idx])
        
        # Apply the transform if specified
        if self.transform:
            sample['depth_gt'] = self.transform(sample['depth_gt'])
            sample['depth_img'] = self.transform(sample['depth_img'])
            sample['image_size'] = sample['depth_gt'].shape
            
        return sample