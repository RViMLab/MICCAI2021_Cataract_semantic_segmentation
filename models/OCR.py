import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Projector import Projector
from torchvision.models import resnet101, resnet50, resnet18, resnet34
from torchvision.models._utils import IntermediateLayerGetter
from utils import CLASS_INFO


class OCRNet(nn.Module):
    eligible_backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet101']

    # Illustration of OCRNet architecture
    #                                               [General]
    # Backbone - layer1 - layer2 - (layer3) - (layer4) -- conv-bn-relu -- conv-bn-relu --->--{
    #                                  |                                                      [OCR]-> conv(cls)
    #                                  L -- conv-bn-relu -- conv(cls) --------->(m)------->--{              |
    #                                                                            |                       upsample
    #                                                                         upsample                      |
    #                                                                            |                         loss
    #                                                                          loss
    #
    #                                                 [OCR]
    #
    #                                                                Q,K,V Transform
    #      x -->                                     (B,C,H,W) x ---> {Q}   SpatialOCR -- x_object_contextual (B,C,H,W)
    #            {Spatial Gather} - global object representation ---> {K,V}
    #      m -->

    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.backbone_name = config['backbone'] if 'backbone' in config else 'resnet101'
        assert (self.backbone_name in self.eligible_backbones), 'backbone must be in {}'.format(self.eligible_backbones)
        self.out_stride = config['out_stride'] if 'out_stride' in config else 8

        self.norm = config['norm'] if 'norm' in config else nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.align_corners = True
        self.dropout = config['dropout'] if 'dropout' in config else 0.0
        self.num_classes = len(CLASS_INFO[experiment][1]) - 1 if 255 in CLASS_INFO[experiment][1].keys() \
            else len(CLASS_INFO[experiment][1])
        # if true,  forward() returns up intermediate logits, up final logits, else only up final logits are returned
        resnet_pretrained = True if 'pretrained' not in config else config['pretrained']
        self.get_intermediate = True

        if 'resnet' in self.backbone_name:
            resnet_group = 1 if 'resnet50' in self.backbone_name or 'resnet101' in self.backbone_name else 0
            assert(self.out_stride in [8, 16, 32])
            if self.out_stride == 8:
                layer_2_stride, layer_3_stride, layer_4_stride = False, True, True
            elif self.out_stride == 16:
                layer_2_stride, layer_3_stride, layer_4_stride = False, False, True
            else:
                layer_2_stride, layer_3_stride, layer_4_stride = False, False, False
            strides = [False, False, False] if resnet_group == 0 else [layer_2_stride, layer_3_stride, layer_4_stride]
            self.backbone_cutoff = {'layer3': 'low', 'layer4': 'high'}
            resnet_class = globals()[self.backbone_name]
            self.backbone = IntermediateLayerGetter(resnet_class(pretrained=resnet_pretrained,
                                                                 replace_stride_with_dilation=strides),
                                                    return_layers=self.backbone_cutoff)
            if resnet_group == 1:
                self.high_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
                self.low_level_channels = self.backbone['layer3']._modules['2'].conv3.out_channels
            else:  # case of resnet 18 or 34: blocks have 2 modules and 2 convs in each
                self.high_out_channels = self.backbone['layer4']._modules['1'].conv2.out_channels
                self.low_level_channels = self.backbone['layer3']._modules['1'].conv2.out_channels
        else:
            raise NotImplementedError('HRNet not yet implemented')

        # maps layer4 features to 512 channels
        self.conv_high_map = nn.Sequential(
            nn.Conv2d(self.high_out_channels, 512, kernel_size=3, stride=1, padding=1),
            self.norm(512),
            self.relu
        )

        # maps layer3 features to intermediate logits

        s = 1 if resnet_group == 1 else 2
        s = 1 if not ('resnet50' in self.backbone_name and self.out_stride == 32) else s

        self.interm_prediction_head = nn.Sequential(
            nn.Conv2d(self.low_level_channels, 512, kernel_size=3, stride=s, padding=1),
            self.norm(512),
            self.relu,
            nn.Dropout2d(self.dropout),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        # ocr
        self.spatial_gather = SpatialGatherModule(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512, key_channels=256,
                                                  out_channels=512, scale=1, dropout=self.dropout,
                                                  norm=self.norm, align_corners=self.align_corners)
        # output
        self.conv_out = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, bias=True)

        # Make projector, if applicable
        if 'projector' in config:
            self.config['projector']['c_in'] = self.high_out_channels
            self.projector_model = Projector(config=self.config['projector'])
            print('added projection from {} to {}'.format(self.high_out_channels, self.projector_model.d))
        else:
            self.projector_model = None

    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)

        backbone_features = self.backbone(x)
        # print('high_features {}'.format(backbone_features['high'].size()))
        # print('low_features {}'.format(backbone_features['low'].size()))

        intermediate_logits = self.interm_prediction_head(backbone_features['low'])
        # print('intermediate_logits {}'.format(intermediate_logits.size()))

        x = self.conv_high_map(backbone_features['high'])
        # print('x {}'.format(x.size()))

        object_global_representation = self.spatial_gather(x, intermediate_logits)
        # print('object_global_representation {}'.format(object_global_representation.size()))

        ocr_representation = self.spatial_ocr_head(x, object_global_representation)
        # print('ocr_representation {}'.format(ocr_representation.size()))

        logits = self.conv_out(ocr_representation)
        # print('logits {}'.format(logits.size()))
        up_logits = F.interpolate(logits, size=input_resolution, mode='bilinear', align_corners=self.align_corners)
        if self.get_intermediate:
            interm_up_logits = F.interpolate(intermediate_logits, size=input_resolution, mode='bilinear',
                                             align_corners=self.align_corners)
            if self.projector_model:
                proj_features = self.projector_model(backbone_features['high'])
                return interm_up_logits, up_logits, proj_features
            else:
                return interm_up_logits, up_logits
        else:
            return up_logits

    def print_params(self):
        # just for debugging
        for w in self.state_dict():
            print(w, "\t", self.state_dict()[w].size())


class SpatialGatherModule(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.cls_num = cls_num  # K:=cls_num
        self.scale = scale

    def forward(self, feats, probs):
        # probs: B, K, H, W  and feats: B, C, H, W
        batch_size = probs.size(0)

        probs = probs.view(batch_size, self.cls_num, -1)  # B, K, N , N:=H*W
        feats = feats.view(batch_size, feats.size(1), -1)  # B, C, N
        feats = feats.permute(0, 2, 1)  # B, N, C
        probs = F.softmax(self.scale * probs, dim=2)  # B, K, N

        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)
        # B, K, N * B, N, C = B, K, C then B, C, K, 1
        return ocr_context


class ObjectAttentionBlock2D(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 norm=nn.BatchNorm2d,
                 aling_corners=True):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.relu = nn.ReLU(inplace=True)
        self.norm = norm

        # φ
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu,
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu
        )

        # ψ
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu,
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu
        )

        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.in_channels),
            self.relu
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # conv of the backbone features to map to key_channels
        # in_channels := C
        # key_channels := C_key
        # x: B, C, H, W
        # proxy: B, C, C_cls, 1 # SpatialGather(x, prediction)

        # φ()
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        # x : B, C, H, W
        # query : B, C_key, H * W | N := H*W
        query = query.permute(0, 2, 1)
        # query : B, N, C_key

        # ψ()
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        # proxy: B, C, C_cls, 1
        # key: B, C_key, C_cls

        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        # proxy: B, C, C_cls, 1
        # value: B, C_key, C_cls
        value = value.permute(0, 2, 1)
        # value: B, C_cls, C_key

        sim_map = torch.matmul(query, key)
        # B, N, C_key * B, C_key, C_cls = B, N, C_cls
        # norm, softmax
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # along the C_cls dim
        # sim_map = B, N, C_cls

        # add bg context ...
        context = torch.matmul(sim_map, value)
        # context =  B, N, C_cls * B, C_cls, C_key = B, N, C_key
        context = context.permute(0, 2, 1).contiguous()
        # context = B, C_key, N
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        # context = B, C_key, H, W
        context = self.f_up(context)
        # context = B, C, H, W
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.0,
                 norm=nn.BatchNorm2d,
                 align_corners=True):
        super(SpatialOCR_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           norm,
                                                           align_corners)
        _in_channels = 2 * in_channels
        # augmented representation: concatenate feat and ocr_feats and conv(1x1,512)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            norm(out_channels),
            self.relu,
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


if __name__ == '__main__':
    config = dict()
    config.update({'backbone': 'resnet50'})
    config.update({'out_stride': 16,
                   'projector': {'mlp': [[1, 256, 1]], 'd': 128}})
    a = torch.ones(size=(1, 3, 540, 960))
    model = OCRNet(config, 2)
    # model.print_params()
    interm, end, proj_feats = model.forward(a)
    print('interm:', interm.shape)
    print('end:', end.shape)
    print('proj_feats:', proj_feats.shape)

