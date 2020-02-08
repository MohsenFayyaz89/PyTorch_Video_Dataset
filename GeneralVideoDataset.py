"""
A Simple PyTorch Video Dataset Class for loading videos using PyTorch
Dataloader. This Dataset assumes that video files are Preprocessed
 by being trimmed over time and resizing the frames.


If you find this code useful, please star the repository.
"""

from __future__ import print_function, division

import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RandomCrop(object):
    """Randomly Crop the frames in a clip."""

    def __init__(self, output_size):
        """
            Args:
              output_size (tuple or int): Desired output size. If int, square crop
              is made.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        h, w = clip.size()[2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        clip = clip[:, :, top : top + new_h, left : left + new_w]

        return clip


class GeneralVideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        clips_list_file,
        root_dir,
        channels,
        time_depth,
        x_size,
        y_size,
        mean,
        transform=None,
    ):
        """
        Args:
            clips_list_file (string): Path to the clipsList file with labels.
            root_dir (string): Directory with all the videoes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            channels: Number of channels of frames
            time_depth: Number of frames to be loaded in a sample
            x_size, y_size: Dimensions of the frames
            mean: Mean value of the training set videos over each channel
        """
        with open(clips_list_file, "rb") as fp:  # Unpickling
            clips_list_file = pickle.load(fp)

        self.clips_list = clips_list_file
        self.root_dir = root_dir
        self.channels = channels
        self.time_depth = time_depth
        self.x_size = x_size
        self.y_size = y_size
        self.mean = mean
        self.transform = transform

    def __len__(self):
        return len(self.clipsList)

    def read_video(self, video_file):
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        frames = torch.FloatTensor(
            self.channels, self.time_depth, self.x_size, self.y_size
        )
        failed_clip = False
        for f in range(self.time_depth):

            ret, frame = cap.read()
            if ret:
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 0, 1)
                frames[:, f, :, :] = frame

            else:
                print("Skipped!")
                failed_clip = True
                break

        for idx in range(len(self.mean)):
            frames[idx] = (frames[idx] - self.mean[idx]) / self.stddev[idx]

        return frames, failed_clip

    def __getitem__(self, idx):

        video_file = os.path.join(self.root_dir, self.clips_list[idx][0])
        clip, failed_clip = self.read_video(video_file)
        if self.transform:
            clip = self.transform(clip)
        sample = {
            "clip": clip,
            "label": self.clips_list[idx][1],
            "failedClip": failed_clip,
        }

        return sample
