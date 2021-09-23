from torch import nn
import torchvision.models as m


class ResNet(nn.Module):
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


class ResNet18(ResNet):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_resnet = m.resnet18(pretrained=self.pretrained, progress=True)

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4


class ResNet34(ResNet):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_resnet = m.resnet34(pretrained=self.pretrained, progress=True)

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4


class ResNet50(ResNet):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_resnet = m.resnet50(pretrained=self.pretrained, progress=True)

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4


class ResNet101(ResNet):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_resnet = m.resnet101(pretrained=self.pretrained, progress=True)

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
