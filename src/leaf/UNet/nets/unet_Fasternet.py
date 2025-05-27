import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone.Fasternet import FasterNet

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # Upsample and then concatenate
        inputs2_up = self.up(inputs2)
        
        # Handle potential size mismatches
        diffY = inputs1.size()[2] - inputs2_up.size()[2]
        diffX = inputs1.size()[3] - inputs2_up.size()[3]
        
        inputs2_up = F.pad(inputs2_up, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        
        outputs = torch.cat([inputs1, inputs2_up], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class FasterNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(FasterNetBackbone, self).__init__()
        self.model = FasterNet(
            fork_feat=True,
            depths=[1, 2, 8, 2],
            embed_dim=50,
            pretrained=pretrained
        )
        
    def forward(self, x):
        # Get all features from FasterNet
        features = self.model(x)
        
        # We need 5 features for proper UNet architecture
        feat1 = features[0]  # /2  [40, 128, 128]
        feat2 = features[1]  # /4  [80, 64, 64]
        feat3 = features[2]  # /8  [160, 32, 32]
        feat4 = features[3]  # /16 [320, 16, 16]
        feat5 = F.avg_pool2d(feat4, 2)  # /32 [320, 8, 8]
        
        return feat1, feat2, feat3, feat4, feat5

class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, backbone='fasternet'):
        super(Unet, self).__init__()
        
        if backbone == 'fasternet':
            self.backbone = FasterNetBackbone(pretrained=None)
            # Updated based on actual feature sizes
            in_filters = [50, 100, 200, 400, 400]  # Last layer is 320 channels
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        
        # Decoder channels
        out_filters = [48, 96, 128, 256]
        
        # First upsampling
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_filters[4], out_filters[3], kernel_size=3, padding=1),  # 320->512
            nn.ReLU(),
            nn.Conv2d(out_filters[3], out_filters[3], kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Up sampling layers
        self.up_concat4 = unetUp(in_filters[3] + out_filters[3], out_filters[2])  # 320+512=832->256
        self.up_concat3 = unetUp(in_filters[2] + out_filters[2], out_filters[1])  # 160+256=416->128
        self.up_concat2 = unetUp(in_filters[1] + out_filters[1], out_filters[0])  # 80+128=208->64
        
        # Final layer
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        # Get features from backbone
        feat1, feat2, feat3, feat4, feat5 = self.backbone(inputs)
        
        # Upsample and decode
        up5 = self.up5(feat5)
        up4 = self.up_concat4(feat4, up5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        
        return self.final(up2)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True