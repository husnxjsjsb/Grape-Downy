from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 使通道数能被8整除的辅助函数
def make_divisible(value: float, divisor: int, min_value: Optional[float] = None, round_down_protect: bool = True) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

# 简单卷积层
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = []
    padding = (kernel_size - 1) // 2
    conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.append(nn.BatchNorm2d(out_channels))
    if act:
        conv.append(nn.ReLU6())
    return nn.Sequential(*conv)

# MobileNetV4的倒残差模块
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
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

# 主干网络的 MobileNetV4 部分
class MobileNetV4(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV4, self).__init__()
        
        # MNV4HybridLarge 的结构配置
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
        self.classifier = nn.Linear(1280, num_classes)

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
        feat1 = self.conv0(x)
        feat2 = self.layer1(feat1)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)
        x = self.layer5(feat5)
        return [feat1, feat2, feat3, feat4, feat5]

# Unet上采样部分
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        if inputs2.size(2) != inputs1.size(2) or inputs2.size(3) != inputs1.size(3):
            inputs2 = nn.functional.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)), mode='bilinear', align_corners=True)
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

# 整合后的Unet分割模型
class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, backbone='mobilenetv4'):
        super(Unet, self).__init__()
        self.backbone = backbone
        
        if self.backbone == 'mobilenetv4':
            self.mobilenet = MobileNetV4(num_classes=num_classes)
            in_filters = [24, 48, 192, 192, 512]  # 更新以匹配特征通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenetv4.'.format(backbone))
        
        out_filters = [64, 128, 256, 512]

        # 更新通道数配置
        self.up_concat4 = unetUp(in_filters[3] + in_filters[4], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2] + out_filters[3], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1] + out_filters[2], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0] + out_filters[1], out_filters[0])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        # 获取各个特征层
        features = self.mobilenet(inputs)
        feat1, feat2, feat3, feat4, feat5 = features

        # # 输出每个特征层的形状
        # print(f"feat1 shape: {feat1.shape}")
        # print(f"feat2 shape: {feat2.shape}")
        # print(f"feat3 shape: {feat3.shape}")
        # print(f"feat4 shape: {feat4.shape}")
        # print(f"feat5 shape: {feat5.shape}")

        # 自上而下地应用解码路径
        feat5 = nn.functional.interpolate(feat5, size=(feat4.size(2), feat4.size(3)), mode='bilinear', align_corners=True)
        up4 = self.up_concat4(feat4, feat5)
        # print(f"up4 shape after up_concat4: {up4.shape}")

        up3 = self.up_concat3(feat3, up4)
        # print(f"up3 shape after up_concat3: {up3.shape}")

        up2 = self.up_concat2(feat2, up3)
        # print(f"up2 shape after up_concat2: {up2.shape}")

        up1 = self.up_concat1(feat1, up2)
        # print(f"up1 shape after up_concat1: {up1.shape}")

        up1 = self.up_conv(up1)
        # print(f"up1 shape after up_conv: {up1.shape}")

        final = self.final(up1)
        print(f"final shape: {final.shape}")

        return final


    
    def freeze_backbone(self):
        if self.backbone == "mobilenetv4":
            for param in self.mobilenet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "mobilenetv4":
            for param in self.mobilenet.parameters():
                param.requires_grad = True


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample,
                 stride, expand_ratio):
        """An inverted bottleneck block with optional depthwises.
        """
        super(UniversalInvertedBottleneckBlock, self).__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv2d(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_, groups=in_channels, act=False)
        expand_filters = make_divisible(in_channels * expand_ratio, 8)
        self._expand_conv = conv2d(in_channels, expand_filters, kernel_size=1)
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        self._proj_conv = conv2d(expand_filters, out_channels, kernel_size=1, stride=1, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
        x = self._expand_conv(x)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
        x = self._proj_conv(x), 
        return x
