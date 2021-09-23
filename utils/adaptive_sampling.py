from torch.utils.data import Sampler
import pandas as pd
import numpy as np
import torch
from utils import softmax, get_inverse_affine_matrix, rotate, InterpolationMode, CLASS_INFO, DEFAULT_VALUES


class AdaptiveBatchSampler(Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, dataframe: pd.DataFrame, iou_values: torch.Tensor,
                 dist_type: str = None, batch_size: int = None, sel_size: int = None):
        super().__init__(data_source=data_source)
        self.data_source = data_source
        self.dataframe = dataframe
        self.iou_values = iou_values  # IoU values per class
        self.dist_type = '1/' if dist_type is None else dist_type
        self.batch_size = 1 if batch_size is None else batch_size
        self.sel_size = 10 if sel_size is None else sel_size

    def get_prob(self):
        prob = None
        if self.dist_type == '1/':
            iou = np.copy(self.iou_values)
            iou[iou > 0] = iou[iou > 0] ** -1
            prob = softmax(iou)
        elif self.dist_type == '1-':
            prob = softmax(1 - np.copy(self.iou_values))
        elif self.dist_type == '1-**2':
            prob = softmax((1 - np.copy(self.iou_values))**2)
        else:
            KeyError("Dataloader AdaptiveBatchSampler: dist_type '{}' not recognised".format(self.dist_type))
        return prob

    def get_dist(self, prob):
        ind = np.argsort(prob)[::-1]
        # print(" - Sampler loop index order: {}".format(ind), end='', flush=True)
        nums = self.batch_size * prob
        sel_nums = np.zeros_like(prob, 'i')
        cum_sum = 0
        for i in ind:  # step through the probabilities in descending order
            to_allocate = self.batch_size - cum_sum  # Find out how many images are left to allocate
            n = int(np.minimum(to_allocate, np.ceil(nums[i])))  # number of images is p * batch_size or to_allocate
            sel_nums[i] = n  # save allocated number
            cum_sum += n  # increase the count of allocated images number
            if cum_sum == self.batch_size:  # if allocated as many as batch_size, stop
                break
        return sel_nums

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        while num_batches > 0:
            prob = self.get_prob()  # Wanted probabilities of priority classes
            dist = self.get_dist(prob)  # Number of images to get as priority from each class
            idx = []
            for i, d in enumerate(dist):
                if d > 0:
                    ind = np.random.choice(range(len(self.data_source)), size=d * self.sel_size, replace=False)
                    ind = np.min(ind.reshape(d, -1), axis=1)
                    idx.extend([*self.dataframe.sort_values(by=i, axis=0, ascending=False)
                               .reset_index().iloc[ind]['level_0']])
            yield idx
            num_batches -= 1

    def __len__(self):
        return len(self.data_source) // self.batch_size
