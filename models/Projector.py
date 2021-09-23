import torch
from torch import nn


class Projector(nn.Module):

    def __init__(self, config):
        """ module that maps encoder features to a d-dimensional space
            if can be a single conv-linear (optionally) preceded by an fcn with conv-relu layers
         """
        super().__init__()
        self.d = config['d'] if 'd' in config else 128  # projection dim
        self.c_in = config['c_in']  # input features channels (usually == output channels of resnet backbone)
        # config['mlp'] list of [k,c] for Conv-Relu layers, if empty only applies Conv(c_in, d, k=1)
        self.mlp = config['mlp'] if 'mlp' in config else []
        self.use_bn = config['use_bn'] if 'use_bn' in config else False

        # sanity checks
        assert(isinstance(self.mlp, list)), 'config["mlp"] must be [[k_1, c_1, s_1], ., [k_n, c_n, s_n]] or [] ' \
                                            'k_i is kernel (k_i x k_i) c_i is channels and s_i is stride'
        if len(self.mlp):
            for layer in self.mlp:
                assert(isinstance(layer, list) and (layer[0] < layer[1])), 'kernel size is first element of list,' \
                                                                            ' got {} {}'.format(layer[0], layer[1])
                assert(len(layer) == 3 and layer[2] in [1, 2]), 'must provide list of lists of 3 elements each' \
                                                                '[kernel, channels, stride] instead {}'.format(layer[2])
        self.convs = []
        c_prev = self.c_in
        if len(self.mlp):
            for (k, c_out, s) in self.mlp:
                print('Projector creating conv layer, k_{}/c_{}/s_{}'.format(k, c_out, s))
                # if use_bn --> do not use bias
                # p = (k + (k - 1) * (d - 1) - s + 1) // 2
                p = (k - s + 1) // 2
                self.convs.append(nn.Conv2d(c_prev, c_out, kernel_size=k, stride=s,
                                            padding=p, bias=not self.use_bn))
                self.convs.append(nn.ReLU(inplace=True))
                if self.use_bn:
                    self.convs.append(nn.BatchNorm2d(c_out, momentum=0.0003))
                c_prev = c_out
        self.convs.append(nn.Conv2d(c_prev, self.d, kernel_size=1, stride=1))
        self.project = nn.Sequential(*self.convs)

    def forward(self, x):
        # # x are features of shape NCHW
        # x = x / torch.norm(x, dim=1, keepdim=True)  # Normalise along C: features vectors now lie on unit hypersphere
        x = self.project(x)
        return x


if __name__ == '__main__':
    # example
    feats = torch.rand(size=(2, 1024, 60, 120)).float()
    proj = Projector({'mlp': [[3, 512, 2], [1, 256, 2]], 'c_in': 1024, 'd': 128, 'use_bn': False})
    projected_feats = proj(feats)
    print(projected_feats.shape)
    for v, par in proj.named_parameters():
        if par.requires_grad:
            print(v, par.data.shape, par.requires_grad)
    d = proj.state_dict()
    print(d)
