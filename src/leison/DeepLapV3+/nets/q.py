from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone.xception import xception
from nets.backbone.mobilenetv2 import mobilenetv2
from nets.backbone.mobilenetv3_sim import mobilenet_v3
from efficientnet_pytorch import EfficientNet
from nets.backbone.microsoft_swintransformer import SwinTransformer
from torchvision import models
import torch.nn as nn
from nets.Attention.CBAM import CBAMBlock
from nets.Attention.SE import SEAttention
from nets.Attention.CAA import CAA
from nets.Attention.ECA import EfficientChannelAttention
from nets.Attention.CPCA import CPCA
from nets.Attention.TripletAttention import TripletAttention
from nets.Attention.ShuffleAttention import ShuffleAttention
from nets.Attention.EMCAD import EMCAM

class AttentionFactory:
    @staticmethod
    def get_attention(attention_type, in_planes, **kwargs):
        if attention_type == "cbam":
            return CBAMBlock(in_planes, **kwargs)
        elif attention_type == "se":
            return SEAttention(in_planes, **kwargs)
        elif attention_type == "caa":
            return CAA(in_planes, **kwargs)
        elif attention_type == "eca":
            return EfficientChannelAttention(in_planes, **kwargs)
        elif attention_type == "cpca":
            return CPCA(in_planes, **kwargs)
        elif attention_type == "ta":
            return TripletAttention(in_planes, **kwargs)
        elif attention_type == "sa":
            return ShuffleAttention(in_planes, **kwargs)
        elif attention_type == "emcam":
            return EMCAM(in_planes, **kwargs)
        else:
            return nn.Identity()  # 默认无注意力机制




class VGG16Backbone(nn.Module):
    def __init__(self, pretrained=True, downsample_factor=16):
        super(VGG16Backbone, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())

        # 根据下采样倍数，选择特征层的截取点
        if downsample_factor == 8:
            self.low_level_features = nn.Sequential(*features[:16])  # 获取前16层作为低特征层
            self.high_level_features = nn.Sequential(*features[16:23])  # 获取接下来几层作为高特征层
        elif downsample_factor == 16:
            self.low_level_features = nn.Sequential(*features[:10])  # 获取前10层作为低特征层
            self.high_level_features = nn.Sequential(*features[10:23])  # 获取接下来几层作为高特征层
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        low_level_features = self.low_level_features(x)
        x = self.high_level_features(low_level_features)
        return low_level_features, x





class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet-b0', pretrained=True, downsample_factor=16):
        super(EfficientNetBackbone, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self.downsample_factor = downsample_factor

        # 根据下采样倍数选择不同的 reduction 层
        if downsample_factor == 8:
            self.low_level_layer = 'reduction_2'  # 选择 reduction_2 层
            self.high_level_layer = 'reduction_4'  # 选择 reduction_4 层
        elif downsample_factor == 16:
            self.low_level_layer = 'reduction_1'  # 选择 reduction_1 层
            self.high_level_layer = 'reduction_5'  # 选择 reduction_5 层
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        # 获取EfficientNet的所有特征层
        endpoints = self.model.extract_endpoints(x)
        # 选择合适的层作为 low_level_features 和 x
        low_level_features = endpoints[self.low_level_layer]  # 选取较浅的特征层
        x = endpoints[self.high_level_layer]  # 选取较深的特征层

        return low_level_features, x


class SwinTransformerBackbone(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True, downsample_factor=16):
        super(SwinTransformerBackbone, self).__init__()
        self.model = SwinTransformer(model_name=model_name, pretrained=pretrained)
        self.downsample_factor = downsample_factor

        if downsample_factor == 8:
            self.low_level_layer_idx = 1  # 假设stage1在索引1处
            self.high_level_layer_idx = 3  # 假设stage3在索引3处
        elif downsample_factor == 16:
            self.low_level_layer_idx = 2  # 假设stage2在索引2处
            self.high_level_layer_idx = 4  # 假设stage4在索引4处
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        features = self.model.forward_features(x)

        # 打印特征的结构和长度
        #print(f"Number of features: {len(features)}")
        #for idx, feature in enumerate(features):
            #print(f"Feature {idx}: {feature.shape}")

        if len(features) > self.low_level_layer_idx:
            low_level_features = features[self.low_level_layer_idx]
        else:
            #print(f"Low level layer index {self.low_level_layer_idx} is out of range.")
            low_level_features = features[0]  # 使用第一个特征层作为默认值

        if len(features) > self.high_level_layer_idx:
            x = features[self.high_level_layer_idx]
        else:
            #print(f"High level layer index {self.high_level_layer_idx} is out of range.")
            x = features[-1]  # 使用最后一个特征层作为默认值

        return low_level_features, x

class EfficientNetB2Backbone(nn.Module):
    def __init__(self, pretrained=True, downsample_factor=16):
        super(EfficientNetB2Backbone, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2') if pretrained else EfficientNet.from_name('efficientnet-b2')
        self.downsample_factor = downsample_factor

        # Choose reduction layers based on downsample factor
        if downsample_factor == 8:
            self.low_level_layer = 'reduction_2'  # Choose reduction_2 layer
            self.high_level_layer = 'reduction_4'  # Choose reduction_4 layer
        elif downsample_factor == 16:
            self.low_level_layer = 'reduction_1'  # Choose reduction_1 layer
            self.high_level_layer = 'reduction_5'  # Choose reduction_5 layer
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(downsample_factor))

    def forward(self, x):
        # Extract all feature endpoints from EfficientNet
        endpoints = self.model.extract_endpoints(x)
        # Select appropriate layers for low-level and high-level features
        low_level_features = endpoints[self.low_level_layer]  # Shallow feature layer
        high_level_features = endpoints[self.high_level_layer]  # Deep feature layer

        return low_level_features, high_level_features

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV3, self).__init__()
        from functools import partial

        model = mobilenet_v3(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )



    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)


        #mid_level_features = self.features[4:6](low_level_features)

        # = F.max_pool2d(mid_level_features, 2)

        x = self.features[4:](low_level_features)

        return low_level_features,  x



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
        x = self._proj_conv(x)
        return x

class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

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

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv3", pretrained=True, downsample_factor=16, attention_type=None):
        super(DeepLab, self).__init__()

        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif "mobilenetv2" in backbone:
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif "mobilenetv3" in backbone:
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 160
            low_level_channels = 24
        elif "mobilenetv4" in backbone:
            self.backbone = MobileNetV4(num_classes=num_classes)
            in_channels = 960
            low_level_channels = 24
        elif "efficientnet-b2" in backbone:
            self.backbone = EfficientNetB2Backbone(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 352 if downsample_factor == 16 else 120
            low_level_channels = 24 if downsample_factor == 16 else 32
        elif "swintransformer" in backbone:
            self.backbone = SwinTransformerBackbone(model_name=backbone, pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 768
            low_level_channels = 192
        elif "vgg16" in backbone:
            self.backbone = VGG16Backbone(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 512
            low_level_channels = 128
        else:
            raise ValueError(f'Unsupported backbone - `{backbone}`, Use mobilenet, xception, efficientnet, etc.')

        self.attention_low = AttentionFactory.get_attention(attention_type, low_level_channels)
        self.attention_high = AttentionFactory.get_attention(attention_type, in_channels)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        self.emcad1 = EMCAM(in_channels=in_channels, out_channels=in_channels)
        self.emcad2 = EMCAM(in_channels=low_level_channels, out_channels=low_level_channels)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level_features, x = self.backbone(x)
        low_level_features = self.attention_low(low_level_features)
        x = self.attention_high(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x