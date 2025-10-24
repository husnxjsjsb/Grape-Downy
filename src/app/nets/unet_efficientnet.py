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
    # def fuse_repvgg_block(self):
    #     if self.deploy:
    #         return
    #     print(f"RepConv.fuse_repvgg_block")

    #     self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])  # 3*3卷积 

    #     self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
    #     rbr_1x1_bias = self.rbr_1x1.bias  # ch_out
    #     # 填充[1,1,1,1]表示在左右上下个填充1个单位，即第三四维(h,w)各增加2
    #     weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])  # co*ci*(ks+2)*(ks+2)

    #     # Fuse self.rbr_identity
    #     if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
    #     	# 0*0支路是BatchNorm2d或SyncBatchNorm的前提是out_channels=in_channels，在RepConv的__init__()中可以看到
    #         identity_conv_1x1 = nn.Conv2d(
    #             in_channels=self.in_channels,
    #             out_channels=self.out_channels,
    #             kernel_size=1,
    #             stride=1,
    #             padding=0,
    #             groups=self.groups,
    #             bias=False)  # （co, ci, 1, 1）
    #         identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
    #         identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()  # (co, ci)
    #         identity_conv_1x1.weight.data.fill_(0.0)
    #         identity_conv_1x1.weight.data.fill_diagonal_(1.0)  # 变成一个单位阵，每个元素可看成一个1*1卷积片
    #         identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)  # (co, ci, 1, 1), 现在我们得到了一个0*0卷积

    #         identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)  # 与BN融合
            
    #         bias_identity_expanded = identity_conv_1x1.bias
    #         weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])  # 将每个1*1卷积片pad一圈0，即在第3、4维各加2
    #     else:
    #     	# channels_out不等于channels_in,零矩阵
    #         bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
    #         weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

    #     self.rbr_dense.weight = torch.nn.Parameter(
    #         self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
    #     self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

    #     self.rbr_reparam = self.rbr_dense
    #     self.deploy = True

    #     if self.rbr_identity is not None:
    #         del self.rbr_identity
    #         self.rbr_identity = None

    #     if self.rbr_1x1 is not None:
    #         del self.rbr_1x1
    #         self.rbr_1x1 = None

    #     if self.rbr_dense is not None:
    #         del self.rbr_dense
    #         self.rbr_dense = None
class RepConv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False):
        super(RepConv1, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                         padding, groups=groups, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride,
                          0, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

    def forward(self, x):
        if self.deploy:
            return self.rbr_reparam(x)
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._pad_1x1_to_3x3_tensor(*self._fuse_bn_tensor(self.rbr_1x1))
        if self.rbr_identity is not None:
            kernel_id, bias_id = self._get_identity_kernel_bias()
        else:
            kernel_id = torch.zeros_like(kernel3x3)
            bias_id = torch.zeros_like(bias3x3)
        kernel = kernel3x3 + kernel1x1 + kernel_id
        bias = bias3x3 + bias1x1 + bias_id
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1, bias1x1):
        pad = (self.kernel_size - 1) // 2
        return F.pad(kernel1x1, [pad, pad, pad, pad]), bias1x1

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        conv = branch[0]
        bn = branch[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _get_identity_kernel_bias(self):
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros((self.out_channels, input_dim, self.kernel_size, self.kernel_size),
                                   dtype=torch.float32, device=self.rbr_dense[0].weight.device)
        for i in range(self.out_channels):
            kernel_value[i, i % input_dim, self.padding, self.padding] = 1.0
        running_std = (self.rbr_identity.running_var + self.rbr_identity.eps).sqrt()
        bias = self.rbr_identity.bias - self.rbr_identity.running_mean * self.rbr_identity.weight / running_std
        weight = self.rbr_identity.weight / running_std
        return kernel_value * weight.reshape(-1, 1, 1, 1), bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.in_channels, self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride,
                                     padding=self.padding, groups=self.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.deploy = True
        for attr in ["rbr_dense", "rbr_1x1", "rbr_identity"]:
            if hasattr(self, attr):
                delattr(self, attr)


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = RepConv(in_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        inputs2 = F.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)),
                                mode='bilinear', align_corners=True)
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.relu(self.conv1(outputs))
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

