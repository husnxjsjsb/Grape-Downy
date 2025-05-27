import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone.xception import xception
from nets.backbone.mobilenetv2 import mobilenetv2
from nets.backbone.mobilenetv3 import mobilenet_v3
from nets.backbone.mobilenetv4 import mobilenetv4_conv_small
from nets.backbone.sim_mobilenetv import MobileNetV3_Small
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


# class DeepLab(nn.Module):
#     def __init__(self, num_classes, backbone="mobilenetv3", pretrained=True, downsample_factor=16, attention_type=None):
#         super(DeepLab, self).__init__()

#         if backbone == "xception":
#             self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 2048
#             low_level_channels = 256
#         elif "mobilenetv2" in backbone:
#             self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         elif "mobilenetv3" in backbone:
#             self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 160
#             low_level_channels = 24
#         elif "mobilenetv4" in backbone:
#             self.backbone = MobileNetV4Backbone(downsample_factor=downsample_factor, pretrained=pretrained)
#             in_channels = 320
#             low_level_channels = 24
#         elif 'efficientnet-b0' in backbone:
#             self.backbone = EfficientNetBackbone(model_name='efficientnet-b0', pretrained=pretrained, downsample_factor=downsample_factor)
#             in_channels = 320 if downsample_factor == 16 else 112
#             low_level_channels = 16 if downsample_factor == 16 else 24
#         elif "swintransformer" in backbone:
#             self.backbone = SwinTransformerBackbone(model_name=backbone, pretrained=pretrained,
#                                                     downsample_factor=downsample_factor)
#             in_channels = 768
#             low_level_channels = 192  # 初始值
#         elif "starnet" in backbone:
#             self.backbone = StarNetBackbone( pretrained=pretrained,
#                                                     downsample_factor=downsample_factor)
#             in_channels = 192
#             low_level_channels = 48  # 初始值
#         elif "vgg16" in backbone:
#             self.backbone = VGG16Backbone(pretrained=pretrained, downsample_factor=downsample_factor)
#             in_channels = 512
#             low_level_channels = 256
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

#         self.attention_low = AttentionFactory.get_attention(attention_type, low_level_channels)
#         self.attention_high = AttentionFactory.get_attention(attention_type, in_channels)

#         self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
#         self.shortcut_conv = nn.Sequential(
#             nn.Conv2d(low_level_channels, 48, 1),  # 初始输入通道数
#             nn.BatchNorm2d(48),
#             nn.ReLU(inplace=True)
#         )

#         self.cat_conv = nn.Sequential(
#             nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),

#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Dropout(0.1),
#         )
#         self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
#         self.emcad1 = EMCAM(in_channels=in_channels, out_channels=in_channels)
#         self.emcad2 = EMCAM(in_channels=low_level_channels, out_channels=low_level_channels)

#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
#         low_level_features, x = self.backbone(x)
#         # print("Low level features shape:", low_level_features.shape) 
#         # print("x shape after backbone:", x.shape)

#         # 动态调整 shortcut_conv 的输入通道数
#         if low_level_features.size(1) != self.shortcut_conv[0].in_channels:
#             self.shortcut_conv[0] = nn.Conv2d(low_level_features.size(1), 48, 1).to(low_level_features.device)

#         low_level_features = self.attention_low(low_level_features)
#         x = self.attention_high(x)
#         # print("x after attention:", x.shape)

#         x = self.aspp(x)
#         low_level_features = self.shortcut_conv(low_level_features)

#         x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
#                           align_corners=True)
#         x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
#         x = self.cls_conv(x)
#         x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
#         return x
class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenetv3", pretrained=True, downsample_factor=16,attention_type=None):
        super(DeepLab, self).__init__()

        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif "mobilenetv2"in backbone:
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif "mobilenetv3"in backbone:
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 200
            # mid_level_channels = 40
            low_level_channels = 24
        elif "mobilenetv3_1"in backbone:
            self.backbone = MobileNetV3_Small(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            # mid_level_channels = 40
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
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear',
                          align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x