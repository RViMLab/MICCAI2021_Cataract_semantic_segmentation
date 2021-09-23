import torch
from torch import nn
# noinspection PyUnresolvedReferences
from losses import *


class LossWrapper(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.loss_weightings = config['losses']
        self.device = config['device']
        self.total_loss = None
        self.loss_classes, self.loss_vals = {}, {}
        self.info_string = ''
        for loss_class in self.loss_weightings:
            if loss_class == 'CrossEntropyLoss':
                if config['experiment'] == 2:
                    ignore_ind = 17
                elif config['experiment'] == 3:
                    ignore_ind = 25
                else:
                    ignore_ind = -100
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_ind)
            else:
                loss_fct = globals()[loss_class](config)
            self.loss_classes.update({loss_class: loss_fct})
            self.loss_vals.update({loss_class: 0})
            self.info_string += loss_class + ', '
        self.info_string = self.info_string[:-2]
        self.dc_off = True if 'dc_off_at_epoch' in self.config else False

    def forward(self,
                deep_features: torch.Tensor,
                prediction: torch.Tensor,
                labels: torch.Tensor,
                loss_list: list = None,
                interm_prediction=None,
                epoch=None) -> torch.Tensor:
        self.total_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        # Compile list of losses to be evaluated. If no specific 'loss_list' is passed
        loss_list = list(self.loss_weightings.keys()) if loss_list is None else loss_list
        for loss_class in self.loss_weightings:  # Go through all the losses
            if loss_class in loss_list:  # Check if this loss should be calculated
                if loss_class == 'LovaszSoftmax':
                    if self.dc_off and epoch is not None and epoch < self.config['dc_off_at_epoch']:
                        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
                    else:
                        loss = self.loss_classes[loss_class](prediction, labels)
                elif loss_class == 'DenseContrastiveLoss':
                    if self.dc_off and epoch is not None and epoch >= self.config['dc_off_at_epoch']:
                        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
                    else:
                        loss = self.loss_classes[loss_class](labels, deep_features)
                elif loss_class == 'TwoScaleLoss':
                    loss = self.loss_classes[loss_class](interm_prediction, prediction, labels.long())
                elif loss_class == 'DenseContrastiveLossV2':
                    loss = self.loss_classes[loss_class](labels, deep_features)
                elif loss_class == 'OhemCrossEntropy':
                    loss = self.loss_classes[loss_class](prediction, labels)
                elif loss_class == 'CrossEntropyLoss':
                    loss = self.loss_classes[loss_class](prediction, labels)
                else:
                    print("Error: Loss class '{}' not recognised!".format(loss_class))
                    loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
            else:
                loss = torch.tensor(0.0, dtype=torch.float, device=self.device)

            # Calculate weighted loss

            loss *= self.loss_weightings[loss_class]
            self.loss_vals[loss_class] = loss
            self.total_loss += loss
        return self.total_loss
