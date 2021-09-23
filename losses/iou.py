import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss, _WeightedLoss



# https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/losses/multi/softiou_loss.py
# focal loss https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/losses/multi/focal_loss.py

class IoU(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super.__init__(reduction=reduction)
        self.register_buffer('epsilon', torch.finfo.eps)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        intersection = input.mul(target).sum(dim=[-2, -1])
        union = (input.mul(1-target) + target).sum(dim=[-2, -1])

        return intersection / (union + self.epsilon)


class WeightedIoU(_WeightedLoss):
    def __init__(self, ):
        pass

    def forward(self):
        pass



def IoU(input: torch.FloatTensor, target: torch.FloatTensor, epsilon=torch.finfo(torch.float32).eps):
    intersection = input.mul(target).sum(dim=[-2, -1])
    union = (input.mul(1-target) + target).sum(dim=[-2, -1])

    return intersection / (union + epsilon)

def WeightedIoU(inputs: torch.FloatTensor, targets: torch.FloatTensor, weights: torch.FloatTensor):
    iou = iou(inputs, targets)
    wiou = iou.dot(weights)

    return wiou

