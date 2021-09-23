import torch
from torch import nn
import torch.nn.functional as f
from utils import CLASS_INFO, conv3x3


class UPerNet(nn.Module):
    """UPerNet implementation, see https://github.com/CSAILVision/semantic-segmentation-pytorch"""
    def __init__(self, config, experiment):
        super().__init__()
        self.num_classes = len(CLASS_INFO[experiment][1])
        if experiment in [2, 3]:
            self.num_classes -= 1  # Ignore the 'ignore' class
        self.pool_scales = [1, 2, 3, 6] if 'pool_scales' not in config else config['pool_scales']
        self.in_channels = config['input_channels']
        self.in_scales = config['input_scales']
        self.ppm_num_ch = 512 if 'ppm_num_ch' not in config else config['ppm_num_ch']
        self.fpn_num_ch = 512 if 'fpn_num_ch' not in config else config['fpn_num_ch']
        self.fpn_num_lvl = len(self.in_scales) if 'fpn_num_lvl' not in config else config['fpn_num_lvl']
        self.fpn_num_lvl = max(self.fpn_num_lvl, 1)  # Make sure no more lvls chosen than exist
        self.fpn_num_lvl = min(self.fpn_num_lvl, len(self.in_scales))  # Make sure no more lvls chosen than exist
        self.interpolate_result_up = True if 'interpolate_result_up' not in config else config['interpolate_result_up']

        # Illustration of PPM (Pyramid Pooling Module) architecture
        #
        # enc[-1] ────────────────────────────────────────────────────────────────────────────────────────┐
        #                                                                                                 │
        # enc[-1] -→ adaptPool-scale[0] -→ interpolate back up -→ conv ppm_num_ch -→ BatchNorm -→ ReLU ───┤
        #                                                                                                 │
        # enc[-1] -→ adaptPool-scale[1] -→ interpolate back up -→ conv ppm_num_ch -→ BatchNorm -→ ReLU ───┤
        #                                                                                                 │
        # enc[-1] -→ adaptPool-scale[2] -→ interpolate back up -→ conv ppm_num_ch -→ BatchNorm -→ ReLU ───┤
        #    ┆                ┆                       ┆                  ┆               ┆         ┆      ┆
        #   ...              ...                     ...                ...             ...       ...     ┆
        #                                                                                                 │
        #                                                                                           [CHANNEL CAT]
        #                                                                                                 │
        #                                                                                          'ppm_last_conv'
        #                                                                                                 ↓
        #                                                                                             'Feature'
        #
        # Illustration of FPN (Feature Pyramid Network) architecture
        #
        # enc[-1] ---------→ PPM ---------→ 'Feature'  -→ upsamp ────────────────────────────[CAT]
        #                                       │                                              │
        #                            interpolate up to match                                   │
        #                                       │                                              │
        # enc[-2] -→ conv -→ BN -→ ReLU ──────[ADD]────── conv -→ BN -→ ReLU -→ upsamp ──────[CAT]
        #                                       │                                              │
        #                            interpolate up to match                                   │
        #                                       │                                              │
        # enc[-3] -→ conv -→ BN -→ ReLU ──────[ADD]────── conv -→ BN -→ ReLU -→ upsamp ──────[CAT]
        #                                       │                                              │
        #                            interpolate up to match                                   │
        #                                       │                                              │
        # enc[-4] -→ conv -→ BN -→ ReLU ──────[ADD]────── conv -→ BN -→ ReLU -→ upsamp ──────[CAT]
        #    ┆        ┆      ┆      ┆                      ┆      ┆      ┆        ┆            ┆
        #   ...      ...    ...    ...                    ...    ...    ...      ...           ┆
        #                                                                                      │
        #                                                                             conv_last fpn_num_ch
        #                                                                                      │
        #                                                                                     ReLU
        #                                                                                      │
        #                                                                            conv to seg channels
        #                                                                                      │
        #                                                                         interpolate to original res
        #                                                                                      │
        #                                                                                  PREDICTION

        # PPM Module

        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in self.pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(self.in_channels[-1], self.ppm_num_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.ppm_num_ch),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3(self.in_channels[-1] + len(self.pool_scales) * self.ppm_num_ch,
                                     self.fpn_num_ch, batch_norm=True, relu=True, stride=1)

        # FPN Module
        self.fpn_in = []
        for in_channel in self.in_channels[-self.fpn_num_lvl:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(in_channel, self.fpn_num_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.fpn_num_ch),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(self.fpn_num_lvl - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3(self.fpn_num_ch, self.fpn_num_ch, batch_norm=True, relu=True, stride=1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3(self.fpn_num_lvl * self.fpn_num_ch, self.fpn_num_ch, batch_norm=True, relu=True, stride=1),
            nn.Conv2d(self.fpn_num_ch, self.num_classes, kernel_size=1)
        )

    def forward(self, conv_out):
        """forward method required by Pytorch"""
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for ppm_pool, ppm_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(ppm_conv(nn.functional.interpolate(
                ppm_pool(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        feature = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [feature]
        for i in range(2, self.fpn_num_lvl + 1):
            conv_x = conv_out[-i]
            conv_x = self.fpn_in[-i + 1](conv_x)  # lateral branch

            feature = nn.functional.interpolate(
                feature, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            feature = conv_x + feature

            fpn_feature_list.append(self.fpn_out[-i + 1](feature))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(2, self.fpn_num_lvl + 1):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[-i + 1],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if self.interpolate_result_up:
            x = f.interpolate(x, scale_factor=self.in_scales[-self.fpn_num_lvl], mode='bilinear', align_corners=False)
        return x


# class UPerNet(nn.Module):
#     """UPerNet implementation, see https://github.com/CSAILVision/semantic-segmentation-pytorch"""
#     def __init__(self, config, experiment):
#         super().__init__()
#         self.num_classes = len(CLASS_INFO[experiment][1])
#         if experiment in [2, 3]:
#             self.num_classes -= 1  # Ignore the 'ignore' class
#         self.pool_scales = [1, 2, 3, 6] if 'pool_scales' not in config else config['pool_scales']
#         self.in_channels = config['input_channels']
#         self.in_scales = config['input_scales']
#         self.ppm_num_ch = 512 if 'ppm_num_ch' not in config else config['ppm_num_ch']
#         self.fpn_num_ch = 256 if 'fpn_num_ch' not in config else config['fpn_num_ch']
#
#         # Illustration of PPM (Pyramid Pooling Module) architecture
#         #
#         # enc[-1] ────────────────────────────────────────────────────────────────────────────────────────┐
#         #                                                                                                 │
#         # enc[-1] -→ adaptPool-scale[0] -→ interpolate back up -→ conv ppm_num_ch -→ BatchNorm -→ ReLU ───┤
#         #                                                                                                 │
#         # enc[-1] -→ adaptPool-scale[1] -→ interpolate back up -→ conv ppm_num_ch -→ BatchNorm -→ ReLU ───┤
#         #                                                                                                 │
#         # enc[-1] -→ adaptPool-scale[2] -→ interpolate back up -→ conv ppm_num_ch -→ BatchNorm -→ ReLU ───┤
#         #    ┆                ┆                       ┆                  ┆               ┆         ┆      ┆
#         #   ...              ...                     ...                ...             ...       ...     ┆
#         #                                                                                                 │
#         #                                                                                           [CHANNEL CAT]
#         #                                                                                                 │
#         #                                                                                          'ppm_last_conv'
#         #                                                                                                 ↓
#         #                                                                                             'Feature'
#         #
#         # Illustration of FPN (Feature Pyramid Network) architecture
#         #
#         # enc[-1] ---------→ PPM ---------→ 'Feature'  -→ upsamp ────────────────────────────[CAT]
#         #                                       │                                              │
#         #                            interpolate up to match                                   │
#         #                                       │                                              │
#         # enc[-2] -→ conv -→ BN -→ ReLU ──────[ADD]────── conv -→ BN -→ ReLU -→ upsamp ──────[CAT]
#         #                                       │                                              │
#         #                            interpolate up to match                                   │
#         #                                       │                                              │
#         # enc[-3] -→ conv -→ BN -→ ReLU ──────[ADD]────── conv -→ BN -→ ReLU -→ upsamp ──────[CAT]
#         #                                       │                                              │
#         #                            interpolate up to match                                   │
#         #                                       │                                              │
#         # enc[-4] -→ conv -→ BN -→ ReLU ──────[ADD]────── conv -→ BN -→ ReLU -→ upsamp ──────[CAT]
#         #    ┆        ┆      ┆      ┆                      ┆      ┆      ┆        ┆            ┆
#         #   ...      ...    ...    ...                    ...    ...    ...      ...           ┆
#         #                                                                                      │
#         #                                                                             conv_last fpn_num_ch
#         #                                                                                      │
#         #                                                                                     ReLU
#         #                                                                                      │
#         #                                                                            conv to seg channels
#         #                                                                                      │
#         #                                                                         interpolate to original res
#         #                                                                                      │
#         #                                                                                  PREDICTION
#
#         # PPM Module
#
#         self.ppm_pooling = []
#         self.ppm_conv = []
#         for scale in self.pool_scales:
#             self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
#             self.ppm_conv.append(nn.Sequential(
#                 nn.Conv2d(self.in_channels[-1], self.ppm_num_ch, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(self.ppm_num_ch),
#                 nn.ReLU(inplace=True)
#             ))
#         self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
#         self.ppm_conv = nn.ModuleList(self.ppm_conv)
#         self.ppm_last_conv = conv3x3(self.in_channels[-1] + len(self.pool_scales) * self.ppm_num_ch,
#                                      self.fpn_num_ch, batch_norm=True, relu=True, stride=1)
#
#         # FPN Module
#         self.fpn_in = []
#         for in_channel in self.in_channels[:-1]:   # skip the top layer
#             self.fpn_in.append(nn.Sequential(
#                 nn.Conv2d(in_channel, self.fpn_num_ch, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(self.fpn_num_ch),
#                 nn.ReLU(inplace=True)
#             ))
#         self.fpn_in = nn.ModuleList(self.fpn_in)
#
#         self.fpn_out = []
#         for i in range(len(self.in_channels) - 1):  # skip the top layer
#             self.fpn_out.append(nn.Sequential(
#                 conv3x3(self.fpn_num_ch, self.fpn_num_ch, batch_norm=True, relu=True, stride=1),
#             ))
#         self.fpn_out = nn.ModuleList(self.fpn_out)
#
#         self.conv_last = nn.Sequential(
#             conv3x3(len(self.in_channels) * self.fpn_num_ch, self.fpn_num_ch, batch_norm=True, relu=True, stride=1),
#             nn.Conv2d(self.fpn_num_ch, self.num_classes, kernel_size=1)
#         )
#
#     def forward(self, conv_out):
#         """forward method required by Pytorch"""
#         conv5 = conv_out[-1]
#
#         input_size = conv5.size()
#         ppm_out = [conv5]
#         for ppm_pool, ppm_conv in zip(self.ppm_pooling, self.ppm_conv):
#             ppm_out.append(ppm_conv(nn.functional.interpolate(
#                 ppm_pool(conv5),
#                 (input_size[2], input_size[3]),
#                 mode='bilinear', align_corners=False)))
#         ppm_out = torch.cat(ppm_out, 1)
#         feature = self.ppm_last_conv(ppm_out)
#
#         fpn_feature_list = [feature]
#         for i in reversed(range(len(conv_out) - 1)):
#             conv_x = conv_out[i]
#             conv_x = self.fpn_in[i](conv_x)  # lateral branch
#
#             feature = nn.functional.interpolate(
#                 feature, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
#             feature = conv_x + feature
#
#             fpn_feature_list.append(self.fpn_out[i](feature))
#
#         fpn_feature_list.reverse()  # [P2 - P5]
#         output_size = fpn_feature_list[0].size()[2:]
#         fusion_list = [fpn_feature_list[0]]
#         for i in range(1, len(fpn_feature_list)):
#             fusion_list.append(nn.functional.interpolate(
#                 fpn_feature_list[i],
#                 output_size,
#                 mode='bilinear', align_corners=False))
#         fusion_out = torch.cat(fusion_list, 1)
#         x = self.conv_last(fusion_out)
#         x = f.interpolate(x, scale_factor=self.in_scales[0], mode='bilinear', align_corners=False)
#
#         return x
