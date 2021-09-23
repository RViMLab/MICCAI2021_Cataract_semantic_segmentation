import torch
import torch.nn as nn
from torch.nn.functional import softmax
from utils import CLASS_INFO, to_one_hot


class SoftIoU(nn.Module):
    """Soft IoU from github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/losses/multi/softiou_loss.py"""
    def __init__(self, config: dict):
        super().__init__()
        self.experiment = config['experiment']
        self.num_classes = len(CLASS_INFO[self.experiment][1])
        self.naive = False if 'naive' not in config else config['naive']

    def forward(self, logit: torch.Tensor, target: torch.Tensor):
        # logit => B x Classes x H x W
        # target => B x H x W
        pred = softmax(logit, dim=1)
        target_one_hot = to_one_hot(target, self.num_classes)

        if self.experiment in [2, 3]:  # Experiments 2 and 3: ignore the last ('ignore') class
            target_one_hot = target_one_hot[:, :-1]
            c = self.num_classes - 1  # Actual number of classes used
        else:
            c = self.num_classes

        # Numerator Product
        inter = pred * target_one_hot
        # Sum over all pixels B x C x H x W => C
        inter = inter.transpose(0, 1).contiguous().view(c, -1).sum(1)

        # Denominator
        union = pred + target_one_hot - (pred * target_one_hot)
        # Sum over all pixels B x C x H x W => C
        union = union.transpose(0, 1).contiguous().view(c, -1).sum(1)

        if self.naive:
            loss = torch.mean(inter / union)
        else:
            loss = inter / union
            loss = torch.mean(loss[union != 0])

        return -loss
