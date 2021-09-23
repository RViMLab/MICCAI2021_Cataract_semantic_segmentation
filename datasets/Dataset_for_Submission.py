import pathlib
import cv2
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose
import numpy as np


class DatasetForSubmission(Dataset):
    def __init__(self, dataframe, experiment, transforms_dict):
        self.df = dataframe
        self.experiment = experiment
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])

    def __getitem__(self, item):
        img_name, img_path = self.df.iloc[item].loc['img_name'], self.df.iloc[item].loc['img_path']
        img = cv2.imread(str(pathlib.Path(img_path)))[..., ::-1]
        tmp_lbl = np.zeros_like(img)[..., 0]
        img, _ = self.common_transforms((img, tmp_lbl))
        img_tensor = self.img_transforms(img)
        return img_name, img_tensor

    def __len__(self):
        return len(self.df)
