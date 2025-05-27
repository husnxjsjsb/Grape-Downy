import functools
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from torch.nn import functional as F
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
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
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size, "map_k should be smaller or equal to kernel_size."
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(*self.origin_kernel_shape), requires_grad=True)  
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        self.convmap = nn.Conv2d(
            in_channels=self.num_2d_kernels,
            out_channels=self.num_2d_kernels,
            kernel_size=map_k,
            stride=1,
            padding=map_k // 2,
            groups=self.num_2d_kernels,  # Adjust groups for pointwise-like operation
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)  # Initialize bias
        self.stride = stride
        self.groups = groups
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        # Ensure the weight shape is correct for the convolution
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        # Combine original weights and convmap result
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        # Perform convolution
        return F.conv2d(inputs, kernel, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
class _routing(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)

class CondConv2D(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
        
        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))
        
        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#         # self.caa = CAA(out_size)

#     def forward(self, inputs1, inputs2):
#         inputs2 = self.up(inputs2)
#         inputs2 = F.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)), mode='bilinear', align_corners=True)
#         # inputs1 = self.caa(inputs1)
#         outputs = torch.cat([inputs1, inputs2], 1)  # Concatenate the inputs
#         outputs = self.relu(self.conv1(outputs))
#         outputs = self.relu(self.conv2(outputs))
#         # outputs = self.caa(outputs)
#         return outputs
class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = Conv(dim, ouc, k=1)
 
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
 
    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x
 
    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        # 将 RepConv 替换为 PConv
        self.conv1 = CondConv2D(in_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        # self.caa = CAA(out_size)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        inputs2 = F.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)), mode='bilinear', align_corners=True)
        # inputs1 = self.caa(inputs1)
        outputs = torch.cat([inputs1, inputs2], 1)  # Concatenate the inputs
        outputs = self.relu(self.conv1(outputs))
        # outputs = self.relu(self.conv2(outputs))
        # outputs = self.caa(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, backbone='efficientnetb0'):
        super(Unet, self).__init__()

        if backbone == 'efficientnetb0':
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
            # EfficientNet-B0 各层的通道数
            in_filters = [16, 24, 40, 112, 320]
        else:
            raise ValueError(f'Unsupported backbone - {backbone}, Use efficientnet-b0.')

        # 解码路径的输出通道数
        out_filters = [48, 96, 128, 256]

        # 构建解码路径
        self.up_concat4 = unetUp(in_filters[4] + in_filters[3], out_filters[3])  # 320 + 112 -> 256
        self.up_concat3 = unetUp(in_filters[2] + out_filters[3], out_filters[2])  # 40 + 256 -> 128
        self.up_concat2 = unetUp(in_filters[1] + out_filters[2], out_filters[1])  # 24 + 128 -> 96
        self.up_concat1 = unetUp(in_filters[0] + out_filters[1], out_filters[0])  # 16 + 96 -> 48

        # 最终上采样卷积
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 最终输出层
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        # 提取 EfficientNet 的特征图
        endpoints = self.efficientnet.extract_endpoints(inputs)

        # 提取不同层次的特征图
        feat1 = endpoints['reduction_1']  # 16 channels
        feat2 = endpoints['reduction_2']  # 24 channels
        feat3 = endpoints['reduction_3']  # 40 channels
        feat4 = endpoints['reduction_4']  # 112 channels
        feat5 = endpoints['reduction_5']  # 320 channels

        # 解码路径：逐层上采样
        up4 = self.up_concat4(feat4, feat5)  # 112 + 320 -> 256
        up3 = self.up_concat3(feat3, up4)    # 40 + 256 -> 128
        up2 = self.up_concat2(feat2, up3)    # 24 + 128 -> 96
        up1 = self.up_concat1(feat1, up2)    # 16 + 96 -> 48

        # 最终上采样和输出
        up1 = self.up_conv(up1)
        final = self.final(up1)

        return final

    def freeze_backbone(self):
        """冻结主干网络的参数"""
        for param in self.efficientnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻主干网络的参数"""
        for param in self.efficientnet.parameters():
            param.requires_grad = True

