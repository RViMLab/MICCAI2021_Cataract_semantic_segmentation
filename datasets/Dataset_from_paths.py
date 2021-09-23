import pathlib
import cv2
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose
from utils import CLASS_INFO, remap_mask


class DatasetFromPath(Dataset):
    def __init__(self, dataframe, data_path, experiment, img_transforms, lbl_transforms, return_ind=None):
        self.df = dataframe
        self.data_path = data_path
        self.experiment = experiment
        self.img_transforms = Compose(img_transforms)
        self.lbl_transforms = Compose(lbl_transforms)
        self.return_ind = False if return_ind is None else return_ind

    def __getitem__(self, item):
        img = cv2.imread(str(pathlib.Path(self.data_path) / self.df.iloc[item].loc['img_path']))[..., ::-1]
        lbl = cv2.imread(str(pathlib.Path(self.data_path) / self.df.iloc[item].loc['lbl_path']), 0)
        lbl = remap_mask(lbl, CLASS_INFO[self.experiment][0], to_network=True).astype('i')
        # Note: .astype('i') is VERY important. If left in uint8, ToTensor() will normalise the segmentation classes!

        # Here (and before Compose(lbl_transforms) we'd need to set the random seed and pray, following this idea:
        # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        # Big yikes. Big potential problem source, see here: https://github.com/pytorch/pytorch/issues/7068
        # If that doesn't work, the whole transforms structure needs to be changed into all-custom functions that will
        # transform both img and lbl at the same time, with one random shift / flip / whatever being applied to both
        img_tensor = self.img_transforms(img)
        lbl_tensor = self.lbl_transforms(lbl).squeeze()
        if self.return_ind:
            return img_tensor, lbl_tensor, item
        return img_tensor, lbl_tensor

    def __len__(self):
        return len(self.df)
