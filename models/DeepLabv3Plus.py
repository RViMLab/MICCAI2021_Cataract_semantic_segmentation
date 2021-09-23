import torch
from torch import nn
from torch.nn import functional as F
from utils.utils import CLASS_INFO
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
from models.Projector import Projector


class DeepLabv3Plus(nn.Module):
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
        # chop off fully connected layers from the backbone + load pre-trained weights
        # we replace stride with dilation in resnet layer4 to make output_stride = 8 (instead of 32)
        self.backbone_cutoff = {'layer1': 'low', 'layer4': 'high'}
        resnet_pretrained = True if 'pretrained' not in config else config['pretrained']
        if self.backbone_name == 'resnet50':
            resnet = resnet50(pretrained=resnet_pretrained, replace_stride_with_dilation=striding)
            self.backbone = IntermediateLayerGetter(resnet, return_layers=self.backbone_cutoff)
            self.high_level_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
            self.low_level_channels = self.backbone['layer1']._modules['2'].conv3.out_channels
        elif self.backbone_name == 'resnet101':
            resnet = resnet101(pretrained=resnet_pretrained, replace_stride_with_dilation=striding)
            self.backbone = IntermediateLayerGetter(resnet, return_layers=self.backbone_cutoff)
            self.high_level_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
            self.low_level_channels = self.backbone['layer1']._modules['2'].conv3.out_channels

        mult = 1 if self.out_stride >= 16 else 2
        # aspp
        self.aspp = ASPP(c_in=self.high_level_channels, c_aspp=self.c_aspp, mult=mult)
        # decoder
        self.decoder = Decoder(self.low_level_channels, self.c_aspp, num_classes=self.num_classes)
        # Make projector, if applicable
        if 'projector' in config:
            self.config['projector']['c_in'] = self.high_level_channels
            self.projector_model = Projector(config=self.config['projector'])
            print('added projection from {} to {}'.format(self.high_level_channels, self.projector_model.d))
        else:
            self.projector_model = None

    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)
        backbone_features = self.backbone(x)  # ['out']
        # print('high_features {}'.format(backbone_features['high'].size()))
        # print('low_features {}'.format(backbone_features['low'].size()))
        aspp_features = self.aspp.forward(backbone_features['high'])
        # print('aspp_features {}'.format(aspp_features.size()))
        logits = self.decoder(backbone_features['low'], aspp_features)
        # print('logits {}:'.format(logits.size()))
        # print('logits {}:'.format(logits.size()))
        upsampled_logits = F.interpolate(logits, size=input_resolution, mode='bilinear', align_corners=True)
        # print('upsampled_logits {}:'.format(upsampled_logits.size()))
        if self.projector_model:
            proj_features = self.projector_model(backbone_features['high'])
            return upsampled_logits, proj_features
        else:
            return upsampled_logits

    def print_params(self):
        # just for debugging
        for w in self.state_dict():
            print(w, "\t", self.state_dict()[w].size())


class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling

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


class Decoder(nn.Module):
    def __init__(self, c_low, c_aspp, num_classes, c_low_reduced=48, c_3x3=256, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003):
        #
        super(Decoder, self).__init__()
        self._c_low = c_low
        self._c_aspp = c_aspp
        self._c_low_reduced = c_low_reduced
        self._c_3x3 = c_3x3
        self.relu = nn.ReLU(inplace=True)

        # conv layer on the low level features
        # padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - stride + 1) // 2  # +1 so // turns into ceil

        self.conv_low = conv(c_low, c_low_reduced, kernel_size=1, stride=1, bias=False)
        self.conv_low_bn = norm(c_low_reduced, momentum)

        # conv layers after concatenation
        self.conv_3x3_1 = conv(c_aspp + c_low_reduced, c_3x3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3x3_1_bn = norm(c_3x3, momentum)

        self.conv_3x3_2 = conv(c_3x3, c_3x3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3x3_2_bn = norm(c_3x3, momentum)

        # output conv
        self.conv_out = nn.Conv2d(c_3x3, num_classes, kernel_size=1, stride=1)

    def forward(self, features_low, features_aspp):
        x1 = self.conv_low(features_low)
        x1 = self.conv_low_bn(x1)
        x1 = self.relu(x1)

        x2 = F.interpolate(features_aspp, size=features_low.size()[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat((x1, x2), 1)  # concatenate along the channel dimension

        x4 = self.conv_3x3_1(x3)
        x4 = self.conv_3x3_1_bn(x4)
        x4 = self.relu(x4)

        x5 = self.conv_3x3_2(x4)
        x5 = self.conv_3x3_2_bn(x5)
        x5 = self.relu(x5)

        x6 = self.conv_out(x5)
        return x6


if __name__ == '__main__':
    config = dict()
    config.update({'backbone': 'resnet50'})
    config.update({'out_stride': 8})
    import pathlib
    checkpoint = torch.load("C:\\Users\\Theodoros Pissas\\PycharmProjects\\pytorch_checkpoints\\ReSim\\resim_c4_backbone_200ep.pth.tar")
    a = torch.ones(size=(1, 3, 540, 960))

    model = DeepLabv3Plus(config, 2)
    # model.print_params()
    b = model.forward(a)
    print(b.shape)
