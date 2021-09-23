import numpy as np
from torch import nn
import torch.nn.functional as f
from utils import CLASS_INFO, padded_conv2d, padded_convtranspose2d


class FCN(nn.Module):
    """Fully convolutional network as per Long et al."""
    def __init__(self, config, experiment):
        super().__init__()
        self.num_classes = len(CLASS_INFO[experiment][1])
        if experiment in [2, 3]:
            self.num_classes -= 1  # Ignore the 'ignore' class
        self.width = config['width']
        n_ch = np.round(np.array([
            64,     # 0 (conv1)
            128,    # 1 (conv2)
            256,    # 2 (conv3)
            512,    # 3 (conv4)
            512,    # 4 (conv5)
            1024,    # 5 (conv6)
            1024,    # 6 (conv7)
        ]) * self.width).astype('i')

        self.conv1 = padded_conv2d(3, n_ch[0], 3)
        self.conv2 = padded_conv2d(n_ch[0], n_ch[1], 3)
        self.conv3 = padded_conv2d(n_ch[1], n_ch[2], 3)
        self.conv4 = padded_conv2d(n_ch[2], n_ch[3], 3)
        self.conv5 = padded_conv2d(n_ch[3], n_ch[4], 3)
        self.conv6 = padded_conv2d(n_ch[4], n_ch[5], 3)
        self.conv7 = padded_conv2d(n_ch[5], n_ch[6], 1)
        self.conv8 = padded_conv2d(n_ch[6], self.num_classes, 1)
        self.p4_conv = padded_conv2d(n_ch[3], self.num_classes, 1)
        self.deconv32 = padded_convtranspose2d(self.num_classes, self.num_classes, 4, stride=2)
        self.p3_conv = padded_conv2d(n_ch[2], self.num_classes, 1)
        self.deconv16 = padded_convtranspose2d(self.num_classes, self.num_classes, 4, stride=2)
        self.deconv8 = padded_convtranspose2d(self.num_classes, self.num_classes, 16, stride=8)

    def forward(self, img):
        """forward method required by Pytorch"""
        c1 = f.relu(self.conv1(img))
        p1 = f.max_pool2d(c1, 2)
        c2 = f.relu(self.conv2(p1))
        p2 = f.max_pool2d(c2, 2)
        c3 = f.relu(self.conv3(p2))
        p3 = f.max_pool2d(c3, 2)
        c4 = f.relu(self.conv4(p3))
        p4 = f.max_pool2d(c4, 2)
        c5 = f.relu(self.conv5(p4))
        p5 = f.max_pool2d(c5, 2)
        c6 = f.relu(self.conv6(p5))
        c7 = f.relu(self.conv7(c6))
        c8 = self.conv8(c7)
        dc32 = self.deconv32(c8)
        p4_c = self.p4_conv(p4)
        fcn_16s = dc32 + p4_c
        dc16 = self.deconv16(fcn_16s)
        p3_c = self.p3_conv(p3)
        fcn_8s = dc16 + p3_c
        fcn = self.deconv8(fcn_8s)
        return fcn
