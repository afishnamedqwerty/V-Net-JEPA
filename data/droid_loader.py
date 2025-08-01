# Droid loader: trajectories, multi-view RGB resize=224x224, action deltas
# data/droid.py
from .loaders import RLDSDataset
import numpy as np
import torch

class DroidDataset(RLDSDataset):
    """Droid RLDS Dataset loader in PyTorch."""
    def __init__(self, data_dir='/data/droid', split='train', **kwargs):
        super().__init__('droid', data_dir, split, **kwargs)

    def __getitem__(self, idx):
        sample = next(self.ds)  # From TFDS numpy iterator
        # Extract RGB video and actions
        # Assume sample structure: {'steps': [{'observation': {'image': array, 'action': array}}]}
        steps = sample['steps']
        frames = []
        actions = []
        for step in steps:
            image = step['observation']['image']  # Assume RGB array
            action = step['action']  # 7D vector
            if self.transform:
                image = self.transform(image)
            frames.append(image)
            actions.append(action)
        video = np.stack(frames)  # T x H x W x C -> ToTensor in transform
        actions = np.stack(actions)  # T x 7
        return torch.from_numpy(video).permute(0, 3, 1, 2).float(), torch.from_numpy(actions).float()  # T x C x H x W, T x 7
