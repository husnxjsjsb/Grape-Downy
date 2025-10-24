import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone.xception import xception
from nets.backbone.mobilenetv2 import mobilenetv2
from nets.backbone.mobilenetv3_sim import mobilenet_v3_sim
from nets.backbone.normal_mobilenetv3 import mobilenet_v3_200
from nets.backbone.mobilenetv4 import mobilenetv4_conv_small
# from nets.backbone.mobilenetv3_1 import MobileNetV3_Small
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
from nets.Starnet import starnet_s1
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
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
    def __init__(self, model_name='efficientnet-b2', pretrained=True, downsample_factor=16):
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
    def __init__(self, downsample_factor=8, pretrained=False):
        super(MobileNetV3, self).__init__()
        from functools import partial

        model = mobilenet_v3_sim(pretrained)
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

class MobileNetV31(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=False):
        super(MobileNetV31, self).__init__()
        from functools import partial

        model = mobilenet_v3_200(pretrained)
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

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        return x * self.se(x)
class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = simam_module(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Small, self).__init__()
        # 修改初始卷积层的输出通道数为24
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.hs1 = act(inplace=True)

        # 修改Block中的通道数
        self.bneck = nn.Sequential(
            Block(3, 24, 24, 24, nn.ReLU, True, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 320, act, True, 1),
            Block(5, 320, 576, 320, act, True, 1),
        )

        # 修改最终卷积层的输出通道数为320
        self.conv2 = nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(320)
        self.hs2 = act(inplace=True)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始卷积
        out = self.hs1(self.bn1(self.conv1(x)))

        # 获取低层特征
        low_level_features = self.bneck[0](out)

        # 处理剩余块
        for block in self.bneck[1:]:
            out = block(out)

        # 最终卷积层，保留四维特征
        out = self.hs2(self.bn2(self.conv2(out)))

        return low_level_features, out  # 返回四维特征
class StarNetBackbone(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=False):
        super(StarNetBackbone, self).__init__()
        self.model = starnet_s1(pretrained=pretrained)
        self.downsample_factor = downsample_factor

    def forward(self, x):
        # 获取 StarNet 的特征层
        features = self.model.stem(x)
        for stage in self.model.stages:
            features = stage(features)
        
        # 根据下采样因子选择合适的特征层
        if self.downsample_factor == 8:
            x_aux = features  # 选择较浅的特征层作为辅助分支输入
            x = features      # 选择较深的特征层作为主干分支输入
        elif self.downsample_factor == 16:
            x_aux = features  # 选择较浅的特征层作为辅助分支输入
            x = features      # 选择较深的特征层作为主干分支输入
        else:
            raise ValueError('Unsupported downsample factor - `{}`, Use 8 or 16.'.format(self.downsample_factor))

        return x_aux, x

class MobileNetV4Backbone(nn.Module):
    def __init__(self, model_name="mobilenetv4_conv_small", pretrained=True, downsample_factor=16):
        super(MobileNetV4Backbone, self).__init__()
        self.model = mobilenetv4_conv_small(num_classes=1000)  # 加载 MobileNetV4 模型
        self.downsample_factor = downsample_factor

        if pretrained:
            # 加载预训练权重（如果有）
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        # 这里可以加载预训练权重
        pass

    def forward(self, x):
        # 获取低层和高层特征
        # print("Input shape:", x.shape) 
        low_level_features, x = self.model(x)
        
        # print("Low level features shape:", low_level_features.shape) 
        # print("x shape after backbone:", x.shape) 
        # # 根据 downsample_factor 调整高层特征的分辨率
        if self.downsample_factor == 16:
            x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        elif self.downsample_factor == 8:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

        return low_level_features, x
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


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv3_1", pretrained=False, downsample_factor=16,attention_type=None):
        super(DeepLab, self).__init__()

        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif "mobilenetv2"in backbone:
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif "sim_mobilenetv3" in backbone:
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 200
            low_level_channels = 24
        elif "n_mobilenetv3"in backbone:
            self.backbone = MobileNetV31(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 200
            low_level_channels = 24
        elif "mobilenetv3_1"in backbone:
            self.backbone = MobileNetV3_Small()
            in_channels = 320
            low_level_channels = 24
        elif "mobilenetv4" in backbone:
            self.backbone = MobileNetV4Backbone(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif 'efficientnet-b0' in backbone:
            self.backbone = EfficientNetBackbone(model_name='efficientnet-b0', pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 320 if downsample_factor == 16 else 112
            low_level_channels = 16 if downsample_factor == 16 else 24
        elif "swintransformer" in backbone:
            self.backbone = SwinTransformerBackbone(model_name=backbone, pretrained=pretrained,
                                                    downsample_factor=downsample_factor)
            in_channels = 768
            low_level_channels = 192  # 初始值
        elif "starnet" in backbone:
            self.backbone = StarNetBackbone( pretrained=pretrained,
                                                    downsample_factor=downsample_factor)
            in_channels = 192
            low_level_channels = 192  # 初始值
        # elif 'efficientnet'in backbone:
        #     self.backbone = EfficientNetBackbone(model_name=backbone, pretrained=pretrained,
        #                                          downsample_factor=downsample_factor)
        #     in_channels = 320 if downsample_factor == 16 else 112  # EfficientNet-B0中 reduction_5 和 reduction_4 的输出通道数
        #     low_level_channels = 16 if downsample_factor == 16 else 24  # EfficientNet-B0中 reduction_1 和 reduction_2 的输出通道数
        elif "swintransformer" in backbone:
            self.backbone = SwinTransformerBackbone(model_name=backbone, pretrained=pretrained,
                                                    downsample_factor=downsample_factor)
            in_channels = 768  # 根据模型的高特征层输出维度设置
            low_level_channels = 192  # 根据模型的低特征层输出维度设置

        elif  "vgg16"in backbone:
            self.backbone = VGG16Backbone(pretrained=pretrained, downsample_factor=downsample_factor)
            in_channels = 512  # VGG16高特征层的输出通道数
            low_level_channels = 128  # VGG16低特征层的输出通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        self.attention_low = AttentionFactory.get_attention(attention_type, low_level_channels)
        self.attention_high = AttentionFactory.get_attention(attention_type, in_channels)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ----------------------------------#
        #   浅层特征边
        # ----------------------------------#
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
        self.emcad1=EMCAM(in_channels=in_channels,out_channels=in_channels)
        self.emcad2=EMCAM(in_channels=low_level_channels,out_channels=low_level_channels)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        # -----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        # -----------------------------------------#
        low_level_features, x = self.backbone(x)
        low_level_features = self.attention_low(low_level_features)
        #low_level_features = self.emcad2(low_level_features)
        x = self.attention_high(x)
        #x = self.emcad1(x)



        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)

        # -----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        # -----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x