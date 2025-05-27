import torch
import torch.nn as nn
from nets.mobilenetv2 import mobilenetv2  # 确保导入你定义的 MobileNetV2 类
from nets.mobilenet import mobilenet_v3

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)  # 拼接后的通道数为 inputs1_channels + inputs2_channels
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, backbone='mobilenetv2'):
        super(Unet, self).__init__()
        if backbone == 'mobilenetv2':
            self.mobilenet = mobilenetv2(pretrained=pretrained)
            self.features = self.mobilenet.features  # 直接引用特征提取部分
            in_filters = [32, 24, 32, 64, 1280]  # backbone输出的特征图通道数
        if backbone == 'mobilenetv3':
            
            self.mobilenet = mobilenet_v3(pretrained=pretrained)
            self.features = self.mobilenet.features  # 直接引用特征提取部分
            in_filters = [32, 24, 32, 64, 176]  # backbone输出的特征图通道数
        if backbone == 'mobilenetv3_1':
            self.mobilenet = mobilenet_v3(pretrained=pretrained)
            self.features = self.mobilenet.features  # 直接引用特征提取部分
            in_filters = [16, 24, 40, 80, 160]  # backbone输出的特征图通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenetv2.'.format(backbone))
        
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3] + in_filters[4], out_filters[3])  # 输入通道数应为64 + 1280
        self.up_concat3 = unetUp(in_filters[2] + out_filters[3], out_filters[2])  # 输入通道数应为32 + 512
        self.up_concat2 = unetUp(in_filters[1] + out_filters[2], out_filters[1])  # 输入通道数应为24 + 256
        self.up_concat1 = unetUp(in_filters[0] + out_filters[1], out_filters[0])  # 输入通道数应为32 + 128

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        x = self.features(inputs)  # 得到单个特征图
        feat1 = self.features[0:1](inputs)  # 示例: 取第一个层
        feat2 = self.features[1:3](feat1)  # 示例: 取第二层
        feat3 = self.features[3:6](feat2)  # 示例: 取第三层
        feat4 = self.features[6:11](feat3)  # 示例: 取第四层
        feat5 = x  # 最后一个特征图

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        up1 = self.up_conv(up1)
        final = self.final(up1)

        return final


    def freeze_backbone(self):
        if self.backbone == "mobilenetv2":
            for param in self.mobilenet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "mobilenetv2":
            for param in self.mobilenet.parameters():
                param.requires_grad = True
