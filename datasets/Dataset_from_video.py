from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import numpy as np
import cv2


class DatasetFromVideo(Dataset):
    """a very non-flexible dataset for reading single frames of videos inspired by mp4 dataset"""
    def __init__(self, video_info: dict, frame_h: int, frame_w: int, get_id=False):

        # assert('capture' in video_info and 'frame_ids' in video_info)
        self.h = frame_h
        self.w = frame_w
        self.video_readers = []
        self.n_frames = []
        self.video_info = video_info
        self.int_to_vid_id = dict()
        self.get_id = get_id

        for int_id, vid_id in enumerate(video_info.keys()):
            self.int_to_vid_id[int_id] = vid_id
            self.video_readers.append(video_info[vid_id]['capture'])
            self.n_frames.append(len(video_info[vid_id]['frame_ids']))

        # self.n_frames = [x - 1 for x in self.n_frames] # subtract one
        # Create index bins, to map accumulated index to video
        self.idx_bins = np.add.accumulate(self.n_frames)

    def __getitem__(self, idx):
        # Map accumulated index to video index
        int_id = np.digitize(idx, self.idx_bins)
        # maps integer video index [0,1,...N] to index according to video file [1,3.. N-1]
        vid_id = self.int_to_vid_id[int_id]

        # for debugging
        if idx == 0:
            self.prev = vid_id
            self.vid_change_idx = []
        if not self.prev == vid_id:
            self.prev = vid_id
            self.vid_change_idx.append(idx)
        # for debugging

        idx = idx % len(self.video_info[vid_id]['frame_ids'])
        frame_idx = self.video_info[vid_id]['frame_ids'][idx]
        self.video_info[vid_id]['capture'].set(1, frame_idx)
        # self.video_info[vid_id]['capture'].set(4, self.w)
        # self.video_info[vid_id]['capture'].set(5, self.h)
        r = self.video_info[vid_id]['capture']
        ret, frame = r.read()
        if not frame.shape[0] == self.h and not frame.shape[1] == self.w:
            frame = cv2.resize(frame, (self.w, self.h))
        # cv2.imshow('frame', frame)
        # bgr to rgb + handling issue with negative strides
        frame = frame[..., ::-1] - np.zeros_like(frame)
        # maps it to [0, 1] and reshapes: H,W,C to C,H,W
        frame = ToTensor()(frame)
        if self.get_id:
            return frame, frame_idx, vid_id
        else:
            return frame

    def __len__(self):
        # Return accumulated length
        return np.array(self.n_frames).sum()


if __name__ == '__main__':
    a = 1
