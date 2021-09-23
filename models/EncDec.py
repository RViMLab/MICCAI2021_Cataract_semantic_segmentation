import torch
from torch import nn
# noinspection PyUnresolvedReferences
from models import *


class EncDec(nn.Module):
    def __init__(self, config, experiment):
        super().__init__()
        """Taken from: https://github.com/CSAILVision/semantic-segmentation-pytorch"""
        self.config = config
        self.experiment = experiment
        enc = globals()[self.config['encoder']['model']]
        dec = globals()[self.config['decoder']['model']]

        # Make encoder, determine channel sizes of encoder outputs
        self.enc_model = enc(config=self.config['encoder'])
        with torch.no_grad():
            tensor_test_dim = 320
            out = self.enc_model(torch.zeros((1, 3, tensor_test_dim, tensor_test_dim), dtype=torch.float))
            channel_sizes = []
            scales = []
            for o in out:
                channel_sizes.append(o.shape[1])
                scales.append(tensor_test_dim // o.shape[2])

        # Update decoder config and make decoder
        self.config['decoder']['input_channels'] = channel_sizes
        self.config['decoder']['input_scales'] = scales
        self.dec_model = dec(config=self.config['decoder'], experiment=self.experiment)

        # Make projector, if applicable
        if 'projector' in self.config:
            self.config['projector']['c_in'] = channel_sizes[-1]
            self.projector_model = Projector(config=self.config['projector'])
        else:
            self.projector_model = None

        # these are added here to enable use with Student Teacher managers
        self.get_features = True
        self.num_classes = self.dec_model.num_classes

    def forward(self, x):
        features = self.enc_model(x)
        prediction = self.dec_model(features)
        if 'projector' in self.config:
            deep_features = self.projector_model(features[-1])
        else:
            deep_features = features[-1]
        if self.get_features:
            return deep_features, prediction  # return only lowest / deepest features
        else:
            return prediction

    def get_num_params(self):
        num_params = sum(p.numel() for p in self.enc_model.parameters() if p.requires_grad)
        if 'projector' in self.config:
            num_params += sum(p.numel() for p in self.projector_model.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in self.dec_model.parameters() if p.requires_grad)
        return num_params
