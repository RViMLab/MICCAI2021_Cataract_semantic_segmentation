import torch
from torchvision.transforms import Compose, Grayscale
from torch.utils.data import Dataset
from decord import VideoReader
from PIL import Image
import numpy as np
import os


# Only supports single workers
# uses decord: https://github.com/dmlc/decord
# other possibilities:
#   NVVL: https://github.com/NVIDIA/nvvl
#   PyAV: https://pyav.org/docs/stable/ (used by torchvision.io https://pytorch.org/docs/stable/torchvision/io.html)
#   OpenCV: VideoCapture, 
#     e.g. capture = cv2.VideoCapture(path)
#          capture.set(1, frame_nr)
#          ret, img = capture.read()
class ColorizationDataset(Dataset):
    def __init__(self, prefix: str, files: list, transforms: Compose = None,
                 sequence_length: int = 1, num_threads: int = 0):
        # Create video reader for each file
        self.video_readers = []
        self.n_frames = []
        for idx, file in enumerate(files):
            absolut_path = os.path.join(prefix, file)
            self.video_readers.append(VideoReader(
                uri=absolut_path,
                num_threads=num_threads
            ))       
            self.n_frames.append(
                len(self.video_readers[idx]) - sequence_length  # substract seq length, to not sample above frame number
            )

        # Create index bins, to map accumulated index to video
        self.idx_bins = np.add.accumulate(self.n_frames)

        # Transforms
        self.transforms = transforms
        self.grayscale = Grayscale(3)
        self.sequence_length = sequence_length

    def __getitem__(self, idx):
        # Map accumulated index to video index
        vid_idx = np.digitize(idx, self.idx_bins)

        # Get frame within video
        frame_idx = self.idx_bins[vid_idx] - idx

        # Sample a sequence in video
        sequence = self.video_readers[vid_idx].get_batch(
            np.arange(frame_idx, frame_idx+self.sequence_length)
        ).asnumpy()

        # Transform sequence with composed transform
        rgb_sequence, gray_sequence = [], []
        if self.transforms:
            for i in range(sequence.shape[0]):
                pil = Image.fromarray(sequence[i])
                rgb_sequence.append(self.transforms(pil))
                gray_sequence.append(self.transforms(self.grayscale(pil)))

        return torch.stack(rgb_sequence), torch.stack(gray_sequence)

    def __len__(self):
        # Return accumulated length
        return np.array(self.n_frames).sum()
