import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import CLASS_INFO


# adapted from https://github.com/HRNet/HRNet-Semantic-Segmentation
class OhemCrossEntropy(nn.Module):
    def __init__(self, config):
        super(OhemCrossEntropy, self).__init__()
        # specify settings through config if they are made explicit else use default
        self.thresh = config['thresh'] if 'thresh' in config else 0.7
        self.min_kept = max(1, config['min_kept']) if 'min_kept' in config else 100000
        if 'experiment' in config:
            self.ignore_label = len(CLASS_INFO[config['experiment']][1]) - 1 if config['experiment'] in [2, 3] else -100
        else:
            self.ignore_label = -100  # if experiment is not given assume nothing is ignored
        self.criterion = nn.CrossEntropyLoss(weight=None,
                                             ignore_index=self.ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()