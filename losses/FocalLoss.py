import torch
import torch.nn as nn
import torch.nn.functional as f


class FocalLoss(nn.Module):
    """Adapted from free-to-use github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py (see license)"""
    def __init__(self, config):
        """

        :param config: dict containing parameters 'gamma' and 'alpha' (optional list of weights)
        """
        super().__init__()
        self.gamma = 2 if 'gamma' not in config else config['gamma']
        if 'alpha' in config:
            self.register_buffer('alpha', torch.tensor(config['alpha']))
        else:
            self.alpha = None

    def forward(self, prediction, target):
        if prediction.dim() > 2:
            prediction = prediction.view(prediction.size(0), prediction.size(1), -1)  # N,C,H,W => N,C,H*W
            prediction = prediction.transpose(1, 2)    # N,C,H*W => N,H*W,C
            prediction = prediction.contiguous().view(-1, prediction.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = f.log_softmax(prediction, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.alpha is not None:
            at = self.alpha.gather(0, target.data.view(-1))
            logpt *= at

        loss = -1 * (1 - pt)**self.gamma * logpt
        return loss.mean()
