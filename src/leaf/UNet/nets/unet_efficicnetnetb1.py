import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
# CoordGate模块
class CoordGate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CoordGate, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取空间位置的坐标信息
        gate = self.conv(x)
        gate = self.sigmoid(gate)  # 通过Sigmoid激活
        return x * gate  # 融合坐标信息与输入特征图

# RefConv模块
class RefConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.cbam = CBAM(out_channels)  # 在 RefConv 中加入 CBAM
        self.coord_gate = CoordGate(out_channels, out_channels)  # 在 RefConv 中加入 CoordGate

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        out = x1 + x3
        out = self.bn(out)
        # out = self.cbam(out)  # 添加 CBAM 注意力模块
        out = self.coord_gate(out)  # 添加 CoordGate 模块来增强空间特征
        return self.relu(out)

# unetUp 模块
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.ref_conv = RefConv(in_size, out_size)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        inputs2 = F.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)), mode='bilinear', align_corners=True)
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.ref_conv(outputs)
        return outputs

# UNet模型
class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, backbone='efficientnetb0'):
        super(Unet, self).__init__()

        if backbone == 'efficientnetb0':
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b2') if pretrained else EfficientNet.from_name('efficientnet-b2')
            in_filters = [16, 24, 48, 120, 352]  # EfficientNet-B2 各层的通道数
        else:
            raise ValueError(f'Unsupported backbone - {backbone}, Use efficientnet-b2.')

        out_filters = [48, 96, 128, 256]

        self.up_concat4 = unetUp(in_filters[4] + in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2] + out_filters[3], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1] + out_filters[2], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0] + out_filters[1], out_filters[0])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            RefConv(out_filters[0], out_filters[0]),
            nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        endpoints = self.efficientnet.extract_endpoints(inputs)

        feat1 = endpoints['reduction_1']
        feat2 = endpoints['reduction_2']
        feat3 = endpoints['reduction_3']
        feat4 = endpoints['reduction_4']
        feat5 = endpoints['reduction_5']

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.up_conv(up1)
        return final

    def freeze_backbone(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = True
