import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Utility function for channel adjustment
def make_divisible(value, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

# Basic convolution layer for MobileNet blocks
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    padding = (kernel_size - 1) // 2
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)]
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)

# Inverted Residual block
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, act=False):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(in_channels * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.append(conv2d(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim),
            conv2d(hidden_dim, out_channels, kernel_size=1, act=act)
        ])
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)

# Universal Inverted Bottleneck Block
class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample, stride, expand_ratio):
        super(UniversalInvertedBottleneckBlock, self).__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            self._start_dw = conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=(stride if not middle_dw_downsample else 1), groups=in_channels)
        expand_filters = make_divisible(in_channels * expand_ratio, 8)
        self._expand_conv = conv2d(in_channels, expand_filters, kernel_size=1)
        if middle_dw_kernel_size:
            self._middle_dw = conv2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=(stride if middle_dw_downsample else 1), groups=expand_filters)
        self._proj_conv = conv2d(expand_filters, out_channels, kernel_size=1, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw(x)
        x = self._expand_conv(x)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
        return self._proj_conv(x)

# MobileNetV4 with block_specs
class MobileNetV4(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV4, self).__init__()
        
        self.block_specs = {
            "conv0": {"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 24, 3, 2]]},
            "layer1": {"block_name": "fused_ib", "num_blocks": 1, "block_specs": [[24, 48, 2, 4.0, True]]},
            "layer2": {"block_name": "uib", "num_blocks": 2, "block_specs": [[48, 96, 3, 5, True, 2, 4], [96, 192, 3, 3, True, 1, 4]]},
            "layer3": {"block_name": "uib", "num_blocks": 11, "block_specs": [[192, 192, 3, 0, True, 1, 4]]},
            "layer4": {"block_name": "uib", "num_blocks": 13, "block_specs": [[192, 512, 5, 5, True, 2, 4]]},
            "layer5": {"block_name": "convbn", "num_blocks": 2, "block_specs": [[512, 960, 1, 1], [960, 1280, 1, 1]]}
        }
        
        self.conv0 = self.build_blocks(self.block_specs["conv0"])
        self.layer1 = self.build_blocks(self.block_specs["layer1"])
        self.layer2 = self.build_blocks(self.block_specs["layer2"])
        self.layer3 = self.build_blocks(self.block_specs["layer3"])
        self.layer4 = self.build_blocks(self.block_specs["layer4"])
        self.layer5 = self.build_blocks(self.block_specs["layer5"])

    def build_blocks(self, layer_spec):
        layers = []
        for spec in layer_spec["block_specs"]:
            if layer_spec["block_name"] == "convbn":
                layers.append(conv2d(*spec))
            elif layer_spec["block_name"] == "fused_ib":
                layers.append(InvertedResidual(*spec))
            elif layer_spec["block_name"] == "uib":
                layers.append(UniversalInvertedBottleneckBlock(*spec))
        return nn.Sequential(*layers)

    def forward(self, x):
        low_level_features = self.conv0(x)
        x = self.layer1(low_level_features)
        x = self.layer2(x)
        x = self.layer3(x)
        high_level_features = self.layer4(x)
        return low_level_features, high_level_features

# ASPP module for Deeplab
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 1), nn.BatchNorm2d(dim_out), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, padding=6*rate, dilation=6*rate), nn.BatchNorm2d(dim_out), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, padding=12*rate, dilation=12*rate), nn.BatchNorm2d(dim_out), nn.ReLU())
        self.branch4 = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, padding=18*rate, dilation=18*rate), nn.BatchNorm2d(dim_out), nn.ReLU())
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim_in, dim_out, 1), nn.BatchNorm2d(dim_out), nn.ReLU())
        self.cat_conv = nn.Sequential(nn.Conv2d(dim_out * 5, dim_out, 1), nn.BatchNorm2d(dim_out), nn.ReLU())

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x4.shape[2:], mode='bilinear', align_corners=True)
        return self.cat_conv(torch.cat([x1, x2, x3, x4, x5], dim=1))

# Deeplab with MobileNetV4 as backbone
class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv4", pretrained=True):
        super(DeepLab, self).__init__()
        
        if backbone == "mobilenetv4":
            self.backbone = MobileNetV4(pretrained=pretrained)
            in_channels = 512
            low_level_channels = 24
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")
        
        self.aspp = ASPP(dim_in=in_channels, dim_out=256)
        self.shortcut_conv = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1), nn.BatchNorm2d(48), nn.ReLU())
        
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        x = F.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat([x, low_level_features], dim=1))
        x = self.cls_conv(x)
        
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
