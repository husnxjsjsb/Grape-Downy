import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))  # Attention factor
        return attn_factor * x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.caa = CAA(out_size)  # Apply CAA here

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        inputs2 = F.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)), mode='bilinear', align_corners=True)
        outputs = torch.cat([inputs1, inputs2], 1)  # Concatenate the inputs
        outputs = self.relu(self.conv1(outputs))
        outputs = self.relu(self.conv2(outputs))

        # Apply CAA attention mechanism
        outputs = self.caa(outputs)

        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, backbone='efficientnetb0'):
        super(Unet, self).__init__()

        if backbone == 'efficientnetb0':
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b2') if pretrained else EfficientNet.from_name('efficientnet-b2')
            in_filters = [16, 24, 48, 120, 352]  # EfficientNet-B2 layers' filter sizes
        else:
            raise ValueError(f'Unsupported backbone - {backbone}, Use efficientnet-b2.')

        out_filters = [48, 96, 128, 256]

        # Define upsampling blocks
        self.up_concat4 = unetUp(in_filters[4] + in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2] + out_filters[3], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1] + out_filters[2], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0] + out_filters[1], out_filters[0])

        # Final upsampling convolution
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        endpoints = self.efficientnet.extract_endpoints(inputs)

        feat1 = endpoints['reduction_1']
        feat2 = endpoints['reduction_2']
        feat3 = endpoints['reduction_3']
        feat4 = endpoints['reduction_4']
        feat5 = endpoints['reduction_5']

        # Upsampling process
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        up1 = self.up_conv(up1)
        final = self.final(up1)

        return final

    def freeze_backbone(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.efficientnet.parameters():
            param.requires_grad = True
