from torch import nn
import torchvision.models as m


class Inception(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, return_feature_maps=True):
        conv_out = []
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        conv_out.append(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        conv_out.append(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        conv_out.append(x)
        # N x 768 x 17 x 17
        # aux_defined = self.training and self.aux_logits
        # if aux_defined:
        #     aux = self.AuxLogits(x)
        # else:
        #     aux = None
        # # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        conv_out.append(x)
        # N x 2048 x 8 x 8
        if return_feature_maps:
            return conv_out
        return [x]


class Inceptionv3(Inception):
    def __init__(self, config):
        super().__init__()
        self.pretrained = config['pretrained']
        orig_inception = m.inception_v3(pretrained=self.pretrained, progress=True)

        # take pretrained resnet, except end bit; input dims: N x 3 x 299 x 299
        self.Conv2d_1a_3x3 = orig_inception.Conv2d_1a_3x3  # N x 32 x 149 x 149
        self.Conv2d_2a_3x3 = orig_inception.Conv2d_2a_3x3  # N x 32 x 147 x 147
        self.Conv2d_2b_3x3 = orig_inception.Conv2d_2b_3x3  # N x 64 x 147 x 147
        self.maxpool1 = orig_inception.maxpool1  # N x 64 x 73 x 73
        self.Conv2d_3b_1x1 = orig_inception.Conv2d_3b_1x1  # N x 80 x 73 x 73
        self.Conv2d_4a_3x3 = orig_inception.Conv2d_4a_3x3  # N x 192 x 71 x 71 - CUT HERE
        self.maxpool2 = orig_inception.maxpool2  # N x 192 x 35 x 35
        self.Mixed_5b = orig_inception.Mixed_5b  # N x 256 x 35 x 35
        self.Mixed_5c = orig_inception.Mixed_5c  # N x 288 x 35 x 35
        self.Mixed_5d = orig_inception.Mixed_5d  # N x 288 x 35 x 35 - CUT HERE
        self.Mixed_6a = orig_inception.Mixed_6a  # N x 768 x 17 x 17
        self.Mixed_6b = orig_inception.Mixed_6b  # N x 768 x 17 x 17
        self.Mixed_6c = orig_inception.Mixed_6c  # N x 768 x 17 x 17
        self.Mixed_6d = orig_inception.Mixed_6d  # N x 768 x 17 x 17
        self.Mixed_6e = orig_inception.Mixed_6e  # N x 768 x 17 x 17 - CUT HERE
        # if aux_defined:
        #     aux = self.AuxLogits(x)
        # else:
        #     aux = None
        # # N x 768 x 17 x 17
        self.Mixed_7a = orig_inception.Mixed_7a  # N x 1280 x 8 x 8
        self.Mixed_7b = orig_inception.Mixed_7b  # N x 2048 x 8 x 8
        self.Mixed_7c = orig_inception.Mixed_7c  # N x 2048 x 8 x 8 - CUT HERE
