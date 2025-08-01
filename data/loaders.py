# Dataset classes: SSv2 (torchvision.video, decode webm to RGB), Droid (HDF5 trajectories, RGB + 7D actions)
# data/loaders.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import json
import h5py  # For potential HDF5 access, but primarily use tfds for RLDS

try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
except ImportError:
    tfds = None
    tf = None

class VideoDataset(Dataset):
    """Base class for video datasets."""
    def __init__(self, root_dir, transform=None, fps=12, frame_size=(224, 224)):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.fps = fps
        self.frame_size = frame_size

    def _load_video(self, video_path):
        """Load video frames using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return torch.stack(frames)  # T x C x H x W

class RLDSDataset(Dataset):
    """Base class for RLDS datasets in PyTorch, wrapping TFDS."""
    def __init__(self, dataset_name, data_dir, split='train', buffer_size=1000, transform=None):
        if tfds is None or tf is None:
            raise ImportError("Please install tensorflow and tensorflow_datasets to use RLDS datasets.")
        self.ds = tfds.load(dataset_name, data_dir=data_dir, split=split, shuffle_files=True)
        self.ds = self.ds.shuffle(buffer_size).as_numpy_iterator()
        self.transform = transform

    def __len__(self):
        return self.ds.cardinality().numpy()  # Approximate if infinite

    def __getitem__(self, idx):
        sample = next(self.ds)
        # Process sample (override in subclass)
        return sample

# Utility to create DataLoader
def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
