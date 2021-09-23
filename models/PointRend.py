from torch import nn
from utils.pointrend_utils import *
from utils import CLASS_INFO
import numpy as np
from models import UPerNet


class PointRend(nn.Module):
    def __init__(self, config, experiment):
        super().__init__()
        self.num_classes = len(CLASS_INFO[experiment][1])
        if experiment in [2, 3]:
            self.num_classes -= 1  # Ignore the 'ignore' class
        self.train_num_pts = config['pr_train_num_pts']  # 14^2 = 196 is enough, more does not improve results
        self.oversample_ratio = 3 if 'pr_oversample_ratio' not in config \
            else config['pr_oversample_ratio']  # 2 < k < 5 is OK, 3 in paper
        self.importance_sample_ratio = .75 if 'pr_importance_sample_ratio' not in config \
            else config['pr_importance_sample_ratio']  # 0.75 < beta < 1 is OK, .75 in paper
        self.subdivision_num_pts = config['pr_subdivision_num_pts']  # 7x7 to 224x224: 28^2 = 784
        self.in_channels = config['input_channels']
        self.in_scales = config['input_scales']
        self.fpn_num_lvl = len(self.in_scales) if 'fpn_num_lvl' not in config else config['fpn_num_lvl']
        self.fpn_num_lvl = max(self.fpn_num_lvl, 1)  # Make sure no more lvls chosen than exist
        self.fpn_num_lvl = min(self.fpn_num_lvl, len(self.in_scales))  # Make sure no more lvls chosen than exist
        config['interpolate_result_up'] = False  # Stop UPerNet from interpolating coarse prediction to full resolution

        # Coarse prediction layer
        # Debugging
        # self.conv_coarse1 = conv3x3(self.in_channels[-1], 128, batch_norm=True, relu=True, stride=1)
        # self.conv_coarse2 = conv3x3(128, self.num_classes, batch_norm=True, relu=True, stride=1)
        self.partial_upernet = UPerNet(config, experiment)

        # Point head
        self.point_head = StandardPointHead(config, self.num_classes)

    def forward(self, conv_out):
        # 1) Get coarse prediction: IMPORTANT: assumed same shape as conv_out[-1]
        # Debugging
        # coarse_pred_logits = self.conv_coarse1(conv_out[-1])
        # coarse_pred_logits = self.conv_coarse2(coarse_pred_logits)
        coarse_pred_logits = self.partial_upernet(conv_out)

        if self.training:
            # 2) Get points and uncertainty
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    coarse_pred_logits,
                    calculate_uncertainty,
                    self.train_num_pts,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # CAREFUL!! 'point_coords' are in cv2 mode, i.e. [x, y] or [hor, ver] rather than numpy [ver, hor]
                h = self.in_scales[-self.fpn_num_lvl] * coarse_pred_logits.shape[2]
                w = self.in_scales[-self.fpn_num_lvl] * coarse_pred_logits.shape[3]
                point_indices = (torch.round((point_coords[..., 1] * (h - 1))) * w +
                                 torch.round((point_coords[..., 0] * (w - 1)))).long()  # N * num_pts

            # 3) Retrieve features
            coarse_features = point_sample(coarse_pred_logits, point_coords)  # Contained kwarg align_corners=False

            # 4) Get fine predictions
            point_list = [point_sample(conv, point_coords) for conv in conv_out[::-1]]  # Contained align_corners=False
            fine_grained_features = cat(point_list, dim=1)
            point_logits = self.point_head(fine_grained_features, coarse_features)

            # 5) Put logit prediction at the right point in the image
            seg_logits = f.interpolate(coarse_pred_logits, scale_factor=self.in_scales[-self.fpn_num_lvl],
                                       mode="bilinear", align_corners=False)
            n, c, h, w = seg_logits.shape
            point_indices = point_indices.unsqueeze(1).expand(-1, c, -1)
            pred = seg_logits.reshape(n, c, h * w).scatter_(2, point_indices, point_logits).view(n, c, h, w)
            return point_coords, point_logits, seg_logits, pred
        else:
            seg_logits = coarse_pred_logits.clone()
            for _ in range(int(np.log2(self.in_scales[-self.fpn_num_lvl]))):
                seg_logits = f.interpolate(seg_logits, scale_factor=2, mode="bilinear", align_corners=False)
                uncertainty_map = calculate_uncertainty(seg_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(uncertainty_map,
                                                                                 self.subdivision_num_pts)
                point_list = [point_sample(conv, point_coords) for conv in conv_out[::-1]]  # Had: align_corners=False
                fine_grained_features = cat(point_list, 1)
                coarse_features = point_sample(seg_logits, point_coords)  # Had: align_corners=False
                point_logits = self.point_head(fine_grained_features, coarse_features)

                # put sem seg point predictions to the right places on the upsampled grid.
                n, c, h, w = seg_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, c, -1)
                seg_logits = (seg_logits.reshape(n, c, h * w).scatter_(2, point_indices, point_logits).view(n, c, h, w))
            return seg_logits


class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """
    def __init__(self, config, num_classes):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super().__init__()
        # fmt: off
        self.num_classes = num_classes
        self.fc_dim = 256 if 'ph_fc_dim' not in config else config['ph_fc_dim']
        self.num_fc = 3 if 'ph_num_fc' not in config else config['ph_num_fc']
        self.coarse_pred_each_layer = True if 'ph_coarse_in_each_layer' not in config \
            else config['ph_coarse_in_each_layer']
        input_channels = sum(ch_num for ch_num in config['input_channels'])
        # fmt: on

        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        for k in range(self.num_fc):
            fc = nn.Conv1d(fc_dim_in, self.fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = self.fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        self.predictor = nn.Conv1d(fc_dim_in, self.num_classes, kernel_size=1, stride=1, padding=0)

        # for layer in self.fc_layers:
        #     weight_init.c2_msra_fill(layer)

        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = f.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = cat((x, coarse_features), dim=1)
        return self.predictor(x)
