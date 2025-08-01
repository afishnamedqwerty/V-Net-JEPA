# SSv2 dataset: video decoding, RGB normalization, augmentations (crops, flips)
# # data/ssv2.py
from .loaders import VideoDataset
import os
import pandas as pd

class SSv2Dataset(VideoDataset):
    """Something-Something-V2 Dataset loader."""
    def __init__(self, root_dir='/data/ssv2', split='train', annotations_file='something-something-v2-labels.json', **kwargs):
        super().__init__(root_dir, **kwargs)
        self.split = split
        self.video_dir = os.path.join(root_dir, split)
        self.annotations = self._load_annotations(annotations_file)
        self.video_list = self._get_video_list()

    def _load_annotations(self, ann_file):
        with open(os.path.join(self.root_dir, ann_file), 'r') as f:
            return json.load(f)

    def _get_video_list(self):
        videos = []
        for vid_id in os.listdir(self.video_dir):
            if vid_id.endswith('.webm'):
                label = self.annotations.get(vid_id.split('.')[0], 'unknown')
                videos.append((os.path.join(self.video_dir, vid_id), label))
        return videos

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]
        frames = self._load_video(video_path)  # T x C x H x W
        return frames, label  # For pretraining, ignore label if SSL
