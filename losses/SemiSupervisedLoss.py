import torch.nn as nn
import torch
from utils import CLASS_INFO
from losses import OhemCrossEntropy, LovaszSoftmax, TwoScaleLoss
from torch.nn import CrossEntropyLoss as CrossEntropy


class SemiSupervisedLoss(nn.Module):
    def __init__(self, config):
        """
        loss that applies the same loss term (specified via config) on pseudo labeled and labeled examples
        each loss term is weighted differently
        :param config:
        """
        super(SemiSupervisedLoss, self).__init__()
        lab_loss_class = globals()[config['labeled']['name']]
        ulab_loss_class = globals()[config['unlabeled']['name']]
        self.w_lab = config['labeled']['weight'] if 'weight' in config['labeled'] else 1.0
        self.w_ulab = config['unlabeled']['weight'] if 'weight' in config['unlabeled'] else 1.0
        self.ignore_label = -100  # if experiment is not given assume nothing is ignored
        if 'experiment' in config:
            self.ignore_label = len(CLASS_INFO[config['experiment']][1]) - 1 if config['experiment'] in [2, 3] else -100

        # pass experiment id to constructors of the two losses
        config['labeled'].update({"experiment": config['experiment']})
        config['unlabeled'].update({"experiment": config['experiment']})

        if config['labeled'] == 'CrossEntropy' and config['unlabeled'] == 'Crossentropy':
            self.loss_lab = lab_loss_class(*config['labeled']['args'], ignore_index=self.ignore_label)
            self.loss_ulab = ulab_loss_class(*config['unlabeled']['args'], ignore_index=self.ignore_label)
        # all other losses expect a config

        elif config['labeled']['name'] == config['unlabeled']['name']:
            c = config['labeled']
            self.loss_lab = lab_loss_class(config['labeled'])
            self.loss_ulab = ulab_loss_class(config['unlabeled'])
        else:
            raise NotImplementedError('different losses for labeled {}'
                                      ' and unlabeled {}'.format(config['labeled'], config['unlabeled']))

        print("Labeled   loss {} with weight {}".format(lab_loss_class, self.w_lab))
        print("Unlabeled loss {} with weight {}".format(ulab_loss_class, self.w_ulab))

    def forward(self, logits, targets, run_on_labeled_validation=False):
        if isinstance(logits, list):
            if len(logits) == 2:
                # two scale loss
                logits_interm, logits_final = logits[0], logits[1]
                if run_on_labeled_validation:
                    loss = self.loss_lab(logits_interm, logits_final, targets)
                else:
                    logits_interm_lab, logits_interm_ulab = self.split_batch(logits_interm)
                    logits_final_lab, logits_final_ulab = self.split_batch(logits_final)
                    targets_lab, targets_ulab = self.split_batch(targets)
                    loss_lab = self.loss_lab(logits_interm_lab, logits_final_lab, targets_lab)
                    loss_ulab = self.loss_ulab(logits_interm_ulab, logits_final_ulab, targets_ulab)
                    loss = loss_lab * self.w_lab + loss_ulab * self.w_ulab
            else:
                raise NotImplementedError('currently only two scales are assumed')

        elif torch.is_tensor(logits):
            if run_on_labeled_validation:
                loss = self.loss_lab(logits, targets)
            else:
                logits_lab, logits_ulab = self.split_batch(logits)
                targets_lab, targets_ulab = self.split_batch(targets)
                loss_lab = self.loss_lab(logits_lab, targets_lab)
                loss_ulab = self.loss_lab(logits_ulab, targets_ulab)
                loss = loss_lab * self.w_lab + loss_ulab * self.w_ulab

        else:
            raise ValueError('logits has to be either torch.tensor or a list'
                             ' of torch tensors instead got {}'.format(type(logits)))
        return loss

    def split_batch(self, batch):
        # batch is a (B,...) shaped tensor
        # assumes first half is always labeled and second half is pseudo
        # assumes batch size is multiple of two
        batch_size = batch.size()[0]
        assert(batch_size >= 2), 'batch_size must be great or equal of 2 instead got {}'.format(batch_size)
        labeled_batch = batch[0:batch_size//2]
        unlabeled_batch = batch[(batch_size//2):]
        return labeled_batch, unlabeled_batch
