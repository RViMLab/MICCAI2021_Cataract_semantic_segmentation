from torch import nn
import torchvision.models as m


class WideResNet(nn.Module):
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


class WideResNet50(WideResNet):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_wideresnet = m.wide_resnet50_2(pretrained=self.pretrained, progress=True)

        # take pretrained wideresnet, except AvgPool and FC
        self.conv1 = orig_wideresnet.conv1
        self.bn1 = orig_wideresnet.bn1
        self.relu = orig_wideresnet.relu
        self.maxpool = orig_wideresnet.maxpool
        self.layer1 = orig_wideresnet.layer1
        self.layer2 = orig_wideresnet.layer2
        self.layer3 = orig_wideresnet.layer3
        self.layer4 = orig_wideresnet.layer4


class WideResNet101(WideResNet):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_wideresnet = m.wide_resnet101_2(pretrained=self.pretrained, progress=True)

        # take pretrained wideresnet, except AvgPool and FC
        self.conv1 = orig_wideresnet.conv1
        self.bn1 = orig_wideresnet.bn1
        self.relu = orig_wideresnet.relu
        self.maxpool = orig_wideresnet.maxpool
        self.layer1 = orig_wideresnet.layer1
        self.layer2 = orig_wideresnet.layer2
        self.layer3 = orig_wideresnet.layer3
        self.layer4 = orig_wideresnet.layer4
