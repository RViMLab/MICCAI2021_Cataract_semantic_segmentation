from torch import nn
import torchvision.models as m


class ResNeXt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, return_feature_maps=True):
        conv_out = []

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResNeXt50(ResNeXt):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_resnext = m.resnext50_32x4d(pretrained=self.pretrained, progress=True)

        # take pretrained resnext, except AvgPool and FC
        self.conv1 = orig_resnext.conv1
        self.bn1 = orig_resnext.bn1
        self.relu = orig_resnext.relu
        self.maxpool = orig_resnext.maxpool
        self.layer1 = orig_resnext.layer1
        self.layer2 = orig_resnext.layer2
        self.layer3 = orig_resnext.layer3
        self.layer4 = orig_resnext.layer4


class ResNeXt101(ResNeXt):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_resnext = m.resnext101_32x8d(pretrained=self.pretrained, progress=True)

        # take pretrained resnext, except AvgPool and FC
        self.conv1 = orig_resnext.conv1
        self.bn1 = orig_resnext.bn1
        self.relu = orig_resnext.relu
        self.maxpool = orig_resnext.maxpool
        self.layer1 = orig_resnext.layer1
        self.layer2 = orig_resnext.layer2
        self.layer3 = orig_resnext.layer3
        self.layer4 = orig_resnext.layer4
