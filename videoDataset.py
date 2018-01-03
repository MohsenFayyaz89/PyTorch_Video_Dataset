"""
PyTorch Video Dataset Class for loading videos using PyTorch
Dataloader. This Dataset assumes that video files are Preprocessed
 by being trimmed over time and resizing the frames.

Mohsen Fayyaz __ Sensifai Vision Group
http://www.Sensifai.com

If you find this code useful, please star the repository.
"""

from __future__ import print_function, division
import cv2
import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class RandomCrop(object):
    """Crop randomly the frames in a clip.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

    def __init__(self, output_size):
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

        clip = clip[:, :, top: top + new_h,
               left: left + new_w]

        return clip


class videoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(self, clipsListFile, rootDir, channels, timeDepth, xSize, ySize, mean, transform=None):
        """
		Args:
			clipsList (string): Path to the clipsList file with labels.
			rootDir (string): Directory with all the videoes.
			transform (callable, optional): Optional transform to be applied
				on a sample.
			channels: Number of channels of frames
			timeDepth: Number of frames to be loaded in a sample
			xSize, ySize: Dimensions of the frames
			mean: Mean valuse of the training set videos over each channel
		"""
        with open(clipsListFile, "rb") as fp:   # Unpickling
            clipsList = pickle.load(fp)

        self.clipsList = clipsList
        self.rootDir = rootDir
        self.channels = channels
        self.timeDepth = timeDepth
        self.xSize = xSize
        self.ySize = ySize
        self.mean = mean
        self.transform = transform


    def __len__(self):
        return len(self.clipsList)

    def readVideo(self, videoFile):
        # Open the video file
        cap = cv2.VideoCapture(videoFile)
        # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = torch.FloatTensor(self.channels, self.timeDepth, self.xSize, self.ySize)
        failedClip = False
        for f in range(self.timeDepth):

            ret, frame = cap.read()
            if ret:
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 0, 1)
                frames[:, f, :, :] = frame

            else:
                print("Skipped!")
                failedClip = True
                break

        for c in range(3):
            frames[c] -= self.mean[c]
        frames /= 255
        return frames, failedClip

    def __getitem__(self, idx):

        videoFile = os.path.join(self.rootDir, self.clipsList[idx][0])
        clip, failedClip = self.readVideo(videoFile)
        if self.transform:
            clip = self.transform(clip)
        sample = {'clip': clip, 'label': self.clipsList[idx][1], 'failedClip': failedClip}

        return sample

