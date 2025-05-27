import torch
import torch.nn as nn
from nets.rs50 import resnetrs50  # 确保导入你定义的 ResNet-RS-50 类

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
    def __init__(self, num_classes=3, pretrained=False, backbone='resnetrs50'):
        super(Unet, self).__init__()
        if backbone == 'resnetrs50':
            self.resnet = resnetrs50(pretrained=pretrained)
            in_filters = [64, 256, 512, 1024, 2048]  # backbone输出的特征图通道数
        else:
            raise ValueError('Unsupported backbone - `{}`, Use resnetrs50.'.format(backbone))
        
        out_filters = [64, 128, 256, 512]

        self.up_concat4 = unetUp(in_filters[3] + in_filters[4], out_filters[3])  # 输入通道数应为1024 + 2048
        self.up_concat3 = unetUp(in_filters[2] + out_filters[3], out_filters[2])  # 输入通道数应为512 + 512
        self.up_concat2 = unetUp(in_filters[1] + out_filters[2], out_filters[1])  # 输入通道数应为256 + 256
        self.up_concat1 = unetUp(in_filters[0] + out_filters[1], out_filters[0])  # 输入通道数应为64 + 128

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
        [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        up1 = self.up_conv(up1)
        final = self.final(up1)

        return final


    def freeze_backbone(self):
        if self.backbone == "resnetrs50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "resnetrs50":
            for param in self.resnet.parameters():
                param.requires_grad = True
