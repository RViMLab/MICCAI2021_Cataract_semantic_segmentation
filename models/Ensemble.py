import torch
from torch import nn
from torch.nn import functional as F
from utils.utils import CLASS_INFO
import os
import pathlib
from models import *
from models.EncDec import EncDec
from torchvision.transforms import Normalize


def get_upernet(config, experiment):
    model = EncDec(config, experiment)
    num_train_params = model.get_num_params()
    print("Using encoder '{}' and decoder {}: {} trainable parameters"
          .format(config['encoder']['model'], config['decoder']['model'], num_train_params))
    model.get_features = False  # suppress returning encoder features
    return model


class Ensemble(nn.Module):
    def __init__(self, config: dict, experiment: int):
        """bagging ensemble of a number of weak-learners"""
        "https://discuss.pytorch.org/t/custom-ensemble-approach/52024/55"
        super(Ensemble, self).__init__()
        self.num_classes = len(CLASS_INFO[experiment][1]) - 1 if 255 in CLASS_INFO[experiment][1].keys() \
            else len(CLASS_INFO[experiment][1])
        self.num_models = len(config.keys())
        self.merge_op = config['merge']
        self.members = []
        self.members_names = [] # for logging only
        self.ckpt_files = []
        self.config = config
        print('Ensemble: with members:')
        # assert(self.num_models > 1), 'ensemble must have at least one model instead got {}'
        for model_i in config['members'].keys():
            model_class = globals()[config['members'][model_i]['model']]
            with torch.no_grad():
                # this is a dirty workaround
                # while deeplab, ocr pack all information for the class constructor in the config,
                # upernet does so using the load_model() of the manager
                # we use a function that does that
                # todo harmonize all models to one of the two ways
                if config['members'][model_i]['model'] == 'UPerNet':
                    model = get_upernet(config['members'][model_i], experiment)
                else:
                    model = model_class(config['members'][model_i], experiment)

            member_name = model.__class__.__name__ if model.__class__.__name__ is not 'Sequential' else 'UPerNet'
            print(member_name)
            self.members_names.append(member_name)
            self.ckpt_files.append(config['members'][model_i]['ckpt'])
            if model.__class__.__name__ == 'OCRNet':
                model.get_intermediate = False
            self.members.append(model)

    def forward(self, x):
        assert x.shape[0] == 1, 'batch size must be one for inference with ensemble'
        outputs = []
        x_ = torch.clone(x)  # store un-normed image
        for model_i, model in enumerate(self.members):
            x = torch.clone(x_)  # get back un-normed image
            if self.config['members'][str(model_i+1)]['model'] == 'UPerNet':
                # UPerNet trained with mean subtraction, deeplab, ocr not trained with it.
                x[0] = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x[0])
            outputs.append(nn.Softmax2d()(model(x)))
            del model
        del x_
        output = torch.stack(outputs)
        if self.merge_op == 'mean':
            output = torch.mean(output, dim=0)
        elif self.merge_op == 'max':
            output = torch.max(output, dim=0)
        return output

    def load_pretrained(self, logging_dir_path, device):
        map_location = 'cuda:{}'.format(device)
        ckpt_paths = [logging_dir_path / ckpt_file / 'chkpts' / 'chkpt_best.pt' for ckpt_file in self.ckpt_files]
        for model, ckpt_path in zip(self.members, ckpt_paths):
            checkpoint = torch.load(str(ckpt_path), map_location)
            state_dict = checkpoint['model_state_dict']
            errors = model.load_state_dict(state_dict, strict=False)
            if len(errors) > 0:
                print('could not load the following variables from checkpoint: \n  {}'.format(errors))
                s = model.state_dict()
                for i in state_dict:
                    if i not in s.keys():
                        assert('projector' in i), 'only projector_model variables can be ' \
                                                  'ignored from ckpt instead tried to load{}'.format(i)
            print('==> succesfully loaded ckpt for model {} '.format(model.__class__.__name__))
