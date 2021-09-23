import pathlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Sampler
from utils import DATA_SPLITS, CLASS_INFO, CLASS_NAMES, get_class_info, reverse_one_to_many_mapping


def get_class_repeat_factors_for_experiment(lbl_df: pd.DataFrame, repeat_thresh: float, exp: int,
                                            return_frequencies=False):
    experiment_cls = CLASS_INFO[exp][1]
    exp_mapping = CLASS_INFO[exp][0]
    rev_mapping = reverse_one_to_many_mapping(exp_mapping)
    canonical_cls = CLASS_NAMES[0]
    canonical_num_to_name = reverse_one_to_many_mapping(CLASS_INFO[0][1])
    num_frames = lbl_df.shape[0]

    cls_freqs = dict()
    cls_rfs = dict()

    for c in canonical_cls:
        c_exp = rev_mapping[canonical_num_to_name[c]]  # from canonical cls name to experiment num
        if c_exp not in cls_freqs.keys():
            cls_freqs[c_exp] = 0
        s = lbl_df.loc[lbl_df[c] > 0].shape[0]
        cls_freqs[c_exp] += s / num_frames

    for c_exp in experiment_cls:
        if cls_freqs[c_exp] == 0:
            cls_freqs[c_exp] = repeat_thresh
        cls_rfs[c_exp] = np.maximum(1, np.sqrt(repeat_thresh / cls_freqs[c_exp]))
    cls_freqs = {k: v for k, v in sorted(cls_freqs.items(), reverse=True, key=lambda item: item[1])}
    cls_rfs = {k: v for k, v in sorted(cls_rfs.items(), reverse=True, key=lambda item: item[1])}
    if return_frequencies:
        return cls_freqs, cls_rfs
    else:
        return cls_rfs


def get_image_repeat_factors_for_experiment(lbl_df: pd.DataFrame, cls_rfs: dict, exp: int):
    exp_mapping = CLASS_INFO[exp][0]
    rev_mapping = reverse_one_to_many_mapping(exp_mapping)  # from canonical to experiment classes
    canonical_cls = CLASS_NAMES[0]
    canonical_num_to_name = reverse_one_to_many_mapping(CLASS_INFO[0][1])  # canonical class to num
    img_rfs = []
    inds = []
    for idx, row in lbl_df.iterrows():  # for each frame
        class_repeat_factors_in_frame = []
        for c in canonical_cls:
            if row[c] > 0:
                class_repeat_factors_in_frame.append(cls_rfs[rev_mapping[canonical_num_to_name[c]]])
        img_rfs.append(np.max(class_repeat_factors_in_frame))
        inds.append(idx)
    return inds, img_rfs


class RepeatFactorSampler(Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, dataframe: pd.DataFrame,
                 repeat_thresh: float, experiment: int, split: int, blacklist=True, seed=None):
        """ Computes repeat factors and returns repeat factor sampler
        Note: this sampler always uses shuffling
        :param data_source: a torch dataset object
        :param dataframe: a dataframe with class occurences as columns
        :param repeat_thresh: repeat factor threshold (intuitively: frequency below which rf kicks in)
        :param experiment: experiment id
        :param split: dataset split being used to determine repeat factors for each image in it.
        :param blacklist: whether blackslisting is to be applied
        :param seed: seeding for torch randomization
        :return RepeatFactorSampler object
        """
        super().__init__(data_source=data_source)
        assert(0 <= repeat_thresh < 1 and split in [0, 1, 2, 5])
        seed = 1 if seed is None else seed
        self.seed = int(seed)
        self.shuffle = True  # shuffling is always used with this sampler
        self.split = split
        self.repeat_thresh = repeat_thresh
        df = get_class_info(dataframe, 0, with_name=True)
        if blacklist:  # drop blacklisted
            df = df.drop(df[df['blacklisted'] == 1].index)
            df.reset_index()
        self.class_repeat_factors, self.repeat_factors = \
            self.repeat_factors_class_and_image_level(df, experiment, repeat_thresh, split)
        self._int_part = torch.trunc(self.repeat_factors)
        self._frac_part = self.repeat_factors - self._int_part
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        self.indices = None

    @staticmethod
    def repeat_factors_class_and_image_level(df: pd.DataFrame, experiment: int, repeat_thresh: float,
                                             split: int):
        train_videos = DATA_SPLITS[split][0]
        train_df = df.loc[df['vid_num'].isin(train_videos)]
        train_df = train_df.reset_index()
        # For each class compute the class-level repeat factor: r(c) = max(1, sqrt(t/f(c)) where f(c) is class freq
        class_rfs = get_class_repeat_factors_for_experiment(train_df, repeat_thresh, experiment)
        # For each image I, compute the image-level repeat factor: r(I) = max_{c in I} r(c)
        inds, rfs = get_image_repeat_factors_for_experiment(train_df, class_rfs, experiment)
        return class_rfs, torch.tensor(rfs, dtype=torch.float32)

    def __iter__(self):
        if self.indices is not None:
            indices = torch.tensor(self.indices, dtype=torch.int64)
        else:
            indices = self._get_epoch_indices(self.g)
        ind_left = self.__len__()
        print('Indices generated {}'.format(ind_left))
        while ind_left > 0:
            # each epoch may have a slightly different size due to the stochastic rounding.
            randperm = torch.randperm(len(indices), generator=self.g)  # shuffling
            for item in indices[randperm]:
                yield int(item)
                ind_left -= 1
        self.indices = None

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self._get_epoch_indices(self.g))

    def _get_epoch_indices(self, generator):
        rands = torch.rand(len(self._frac_part), generator=generator)
        rounded_rep_factors = self._int_part + (rands < self._frac_part).float()
        indices = []
        # replicate each image's index by its rounded repeat factor
        for img_index, rep_factor in enumerate(rounded_rep_factors):
            indices.extend([img_index] * int(rep_factor.item()))
        self.indices = indices
        return torch.tensor(indices, dtype=torch.int64)



