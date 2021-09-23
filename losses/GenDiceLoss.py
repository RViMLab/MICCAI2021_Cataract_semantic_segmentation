import torch
import torch.nn as nn
from torch.nn.functional import softmax
from utils import CLASS_INFO, to_one_hot


class GenDiceLoss(nn.Module):
    """Generalised DICE Loss, see Sudre et al, "Generalised Dice overlap as a deep learning loss function for highly
    unbalanced segmentations" (https://arxiv.org/pdf/1707.03237.pdf)"""
    def __init__(self, config: dict):
        super().__init__()
        self.experiment = config['experiment']
        self.num_classes = len(CLASS_INFO[self.experiment][1])
        self.weights = config['weights'] if 'weights' in config else None
        self.naive = False if 'naive' not in config else config['naive']

    def forward(self, logit: torch.Tensor, target: torch.Tensor):
        # logit => B x Classes x H x W
        # target => B x H x W
        pred = softmax(logit, dim=1)
        target_one_hot = to_one_hot(target, self.num_classes)

        if self.experiment in [2, 3]:  # Experiments 2 and 3: ignore the last ('ignore') class
            target_one_hot = target_one_hot[:, :-1]
            c = self.num_classes - 1
        else:
            c = self.num_classes

        dividend = (pred * target_one_hot).transpose(0, 1).sum(1).sum(1).sum(1)
        divisor = (pred + target_one_hot).transpose(0, 1).sum(1).sum(1).sum(1)

        if self.weights is not None:
            if self.weights == 'auto':
                weights = target_one_hot.transpose(0, 1).sum(1).sum(1).sum(1) ** 2
                weights[weights == 0] = 1  # To avoid inf when calculating the inverse
                weights = 1 / weights
            else:
                assert c == len(self.weights), "Number of weights does not match number of logit channels"
                weights = torch.tensor(self.weights).to(logit.device)
            dividend = dividend * weights
            divisor = divisor * weights
        frac = dividend / divisor
        if self.naive:
            mean = torch.mean(frac)
        else:
            mean = torch.mean(frac[divisor != 0])  # Classes that are not present get ignored
        generalised_dice_loss = 1 - 2 * mean
        return generalised_dice_loss
