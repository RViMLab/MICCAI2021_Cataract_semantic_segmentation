import torch
from torch import nn
from torch.nn import functional as F
from utils.utils import CLASS_INFO
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import os
from models.Projector import Projector


class DeepLabv3(nn.Module):
    eligible_backbones = ['resnet50', 'resnet101']

    def __init__(self, config, experiment):
        super().__init__()
        self.backbone_name = config['backbone'] if 'backbone' in config else 'resnet50'
        self.c_aspp = config['aspp']['channels'] if 'aspp' in config else 256
        self.out_stride = config['out_stride'] if 'out_stride' in config else 16
        self.config = config
        assert(self.out_stride in [8, 16, 32])
        if self.out_stride == 8:
            layer_2_stride, layer_3_stride, layer_4_stride = False, True, True
        elif self.out_stride == 16:
            layer_2_stride, layer_3_stride, layer_4_stride = False, False, True
        else:
            layer_2_stride, layer_3_stride, layer_4_stride = True, True, True
        striding = [layer_2_stride, layer_3_stride, layer_4_stride]
        assert(self.backbone_name in self.eligible_backbones), 'backbone must be in {}'.format(self.eligible_backbones)
        self.num_classes = len(CLASS_INFO[experiment][1]) - 1 if 255 in CLASS_INFO[experiment][1].keys() \
            else len(CLASS_INFO[experiment][1])
        # chop off fully connected layers from the backbone + load pretrained weights
        # we replace stride with dilation in resnet layer 4 to make output_stride = 8 (instead of higher by default)
        self.backbone_cutoff = {'layer4': 'out'}
        resnet_pretrained = True if 'pretrained' not in config else config['pretrained']
        if self.backbone_name == 'resnet50':
            self.backbone = IntermediateLayerGetter(resnet50(pretrained=resnet_pretrained,
                                                             replace_stride_with_dilation=striding),
                                                    return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
        elif self.backbone_name == 'resnet101':
            self.backbone = IntermediateLayerGetter(resnet101(pretrained=resnet_pretrained,
                                                              replace_stride_with_dilation=striding),
                                                    return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
        # define Atrous Spatial Pyramid Pooling layer
        mult = 1 if self.out_stride >= 16 else 2
        # Todo change mult = 1 to mult = mult (for now keeping it for backwards compatibility with pretrained)
        self.aspp = ASPP(c_in=self.backbone_out_channels, c_aspp=self.c_aspp, mult=mult)
        self.conv_out = nn.Conv2d(self.c_aspp, self.num_classes, kernel_size=1, stride=1)

        if 'projector' in config:
            self.config['projector']['c_in'] = self.backbone_out_channels
            self.projector_model = Projector(config=self.config['projector'])
            print('added projection from {} to {}'.format(self.backbone_out_channels, self.projector_model.d))
        else:
            self.projector_model = None

    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)
        backbone_features = self.backbone.forward(x)['out']
        # print(backbone_features.size())
        aspp_features = self.aspp.forward(backbone_features)
        logits = self.conv_out(aspp_features)
        # print(logits.size())
        upsampled_logits = F.interpolate(logits, size=input_resolution, mode='bilinear', align_corners=True)

        if self.projector_model:
            proj_features = self.projector_model(backbone_features)
            return upsampled_logits, proj_features
        else:
            return upsampled_logits

    def print_params(self):
        # just for debugging
        for w in self.state_dict():
            print(w, "\t", self.state_dict()[w].size())

    def init_weights(self, pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            #     # logger.info(
            #         # '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class ASPP(nn.Module):

    def __init__(self, c_in, c_aspp, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._c_in = c_in
        self._c_aspp = c_aspp

        # image level features
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(c_in, c_aspp, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(c_in, c_aspp, kernel_size=3, stride=1, dilation=int(6*mult), padding=int(6*mult), bias=False)
        self.aspp3 = conv(c_in, c_aspp, kernel_size=3, stride=1, dilation=int(12*mult), padding=int(12*mult), bias=False)
        self.aspp4 = conv(c_in, c_aspp, kernel_size=3, stride=1, dilation=int(18*mult), padding=int(18*mult), bias=False)
        self.aspp5 = conv(c_in, c_aspp, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(c_aspp, momentum)
        self.aspp2_bn = norm(c_aspp, momentum)
        self.aspp3_bn = norm(c_aspp, momentum)
        self.aspp4_bn = norm(c_aspp, momentum)
        self.aspp5_bn = norm(c_aspp, momentum)
        self.conv2 = conv(c_aspp * 5, c_aspp, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm(c_aspp, momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)  # concatenate along the channel dimension
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    # import pathlib
    # import os
    config = dict()
    config.update({'backbone': 'resnet50'})
    config.update({'out_stride': 16})
    a = torch.ones(size=(2, 3, 540, 960))
    model = DeepLabv3(config, 2)
    # pretrained = r'C:\Users\Theodoros Pissas\Documents\GitHub\logging\DeepLav3\blacklist\20200905_113313_e2__DeepLabv3\ckpts\chkpt_best.pt'
    # print(os.path.isfile(pretrained))
    # model.init_weights(pretrained)
    # # model.print_params()
    # b = model.forward(a)
    # print(b.shape)

