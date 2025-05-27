import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
import numpy as np
import torch
from torch import nn
from torch.nn import init
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

# class CPCA_ChannelAttention(nn.Module):

#     def __init__(self, input_channels, internal_neurons):
#         super(CPCA_ChannelAttention, self).__init__()
#         self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
#                              bias=True)
#         self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
#                              bias=True)
#         self.input_channels = input_channels

#     def forward(self, inputs):
#         x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
#         x1 = self.fc1(x1)
#         x1 = F.relu(x1, inplace=True)
#         x1 = self.fc2(x1)
#         x1 = torch.sigmoid(x1)
#         x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
#         x2 = self.fc1(x2)
#         x2 = F.relu(x2, inplace=True)
#         x2 = self.fc2(x2)
#         x2 = torch.sigmoid(x2)
#         x = x1 + x2
#         x = x.view(-1, self.input_channels, 1, 1)
#         return inputs * x


# class CPCA(nn.Module):
#     def __init__(self, channels, channelAttention_reduce=4):
#         super().__init__()

#         self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
#         self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
#         self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
#         self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
#         self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
#         self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
#         self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
#         self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
#         self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
#         self.act = nn.GELU()

#     def forward(self, inputs):
#         #   Global Perceptron
#         inputs = self.conv(inputs)
#         inputs = self.act(inputs)

#         inputs = self.ca(inputs)

#         x_init = self.dconv5_5(inputs)
#         x_1 = self.dconv1_7(x_init)
#         x_1 = self.dconv7_1(x_1)
#         x_2 = self.dconv1_11(x_init)
#         x_2 = self.dconv11_1(x_2)
#         x_3 = self.dconv1_21(x_init)
#         x_3 = self.dconv21_1(x_3)
#         x = x_1 + x_2 + x_3 + x_init
#         spatial_att = self.conv(x)
#         out = spatial_att * inputs
#         out = self.conv(out)
#         return out

# class ShuffleAttention(nn.Module):

#     def __init__(self, channel=512, reduction=16, G=8):
#         super().__init__()
#         self.G = G
#         self.channel = channel
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
#         self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
#         self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
#         self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
#         self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
#         self.sigmoid = nn.Sigmoid()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape
#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)

#         # flatten
#         x = x.reshape(b, -1, h, w)

#         return x

#     def forward(self, x):
#         b, c, h, w = x.size()
#         # group into subfeatures
#         x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

#         # channel_split
#         x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

#         # channel attention
#         x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
#         x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
#         x_channel = x_0 * self.sigmoid(x_channel)

#         # spatial attention
#         x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
#         x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
#         x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

#         # concatenate along channel axis
#         out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
#         out = out.contiguous().view(b, -1, h, w)

#         # channel shuffle
#         out = self.channel_shuffle(out, 2)
#         return out
# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


# class ZPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# class AttentionGate(nn.Module):
#     def __init__(self):
#         super(AttentionGate, self).__init__()
#         kernel_size = 7
#         self.compress = ZPool()
#         self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.conv(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale


# class TripletAttention(nn.Module):
#     def __init__(self, no_spatial=False):
#         super(TripletAttention, self).__init__()
#         self.cw = AttentionGate()
#         self.hc = AttentionGate()
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.hw = AttentionGate()

#     def forward(self, x):
#         x_perm1 = x.permute(0, 2, 1, 3).contiguous()
#         x_out1 = self.cw(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
#         x_perm2 = x.permute(0, 3, 2, 1).contiguous()
#         x_out2 = self.hc(x_perm2)
#         x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
#         if not self.no_spatial:
#             x_out = self.hw(x)
#             x_out = 1 / 3 * (x_out + x_out11 + x_out21)
#         else:
#             x_out = 1 / 2 * (x_out11 + x_out21)
#         return x_out
# class GroupBatchnorm2d(nn.Module):
#     def __init__(self, c_num:int, 
#                  group_num:int = 16, 
#                  eps:float = 1e-10
#                  ):
#         super(GroupBatchnorm2d,self).__init__()
#         assert c_num    >= group_num
#         self.group_num  = group_num
#         self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
#         self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
#         self.eps        = eps
#     def forward(self, x):
#         N, C, H, W  = x.size()
#         x           = x.view(   N, self.group_num, -1   )
#         mean        = x.mean(   dim = 2, keepdim = True )
#         std         = x.std (   dim = 2, keepdim = True )
#         x           = (x - mean) / (std+self.eps)
#         x           = x.view(N, C, H, W)
#         return x * self.weight + self.bias


# class SRU(nn.Module):
#     def __init__(self,
#                  oup_channels:int, 
#                  group_num:int = 16,
#                  gate_treshold:float = 0.5,
#                  torch_gn:bool = True
#                  ):
#         super().__init__()
        
#         self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
#         self.gate_treshold  = gate_treshold
#         self.sigomid        = nn.Sigmoid()

#     def forward(self,x):
#         gn_x        = self.gn(x)
#         w_gamma     = self.gn.weight/sum(self.gn.weight)
#         w_gamma     = w_gamma.view(1,-1,1,1)
#         reweigts    = self.sigomid( gn_x * w_gamma )
#         # Gate
#         w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
#         w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
#         x_1         = w1 * x
#         x_2         = w2 * x
#         y           = self.reconstruct(x_1,x_2)
#         return y
    
#     def reconstruct(self,x_1,x_2):
#         x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
#         x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
#         return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


# class CRU(nn.Module):
#     '''
#     alpha: 0<alpha<1
#     '''
#     def __init__(self, 
#                  op_channel:int,
#                  alpha:float = 1/2,
#                  squeeze_radio:int = 2 ,
#                  group_size:int = 2,
#                  group_kernel_size:int = 3,
#                  ):
#         super().__init__()
#         self.up_channel     = up_channel   =   int(alpha*op_channel)
#         self.low_channel    = low_channel  =   op_channel-up_channel
#         self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
#         self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
#         #up
#         self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
#         self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
#         #low
#         self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
#         self.advavg         = nn.AdaptiveAvgPool2d(1)

#     def forward(self,x):
#         # Split
#         up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
#         up,low  = self.squeeze1(up),self.squeeze2(low)
#         # Transform
#         Y1      = self.GWC(up) + self.PWC1(up)
#         Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
#         # Fuse
#         out     = torch.cat( [Y1,Y2], dim= 1 )
#         out     = F.softmax( self.advavg(out), dim=1 ) * out
#         out1,out2 = torch.split(out,out.size(1)//2,dim=1)
#         return out1+out2


# class ScConv(nn.Module):
#     def __init__(self,
#                 op_channel:int,
#                 group_num:int = 4,
#                 gate_treshold:float = 0.5,
#                 alpha:float = 1/2,
#                 squeeze_radio:int = 2 ,
#                 group_size:int = 2,
#                 group_kernel_size:int = 3,
#                  ):
#         super().__init__()
#         self.SRU = SRU( op_channel, 
#                        group_num            = group_num,  
#                        gate_treshold        = gate_treshold )
#         self.CRU = CRU( op_channel, 
#                        alpha                = alpha, 
#                        squeeze_radio        = squeeze_radio ,
#                        group_size           = group_size ,
#                        group_kernel_size    = group_kernel_size )
    
#     def forward(self,x):
#         x = self.SRU(x)
#         x = self.CRU(x)
#         return x
class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1, map_k=3):
        super(RepConv, self).__init__()
        assert map_k <= kernel_size, "map_k should be smaller or equal to kernel_size."
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(*self.origin_kernel_shape), requires_grad=True)  # Initialize weight
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

# class Attention(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
#         super(Attention, self).__init__()
#         attention_channel = max(int(in_planes * reduction), min_channel)
#         self.kernel_size = kernel_size
#         self.kernel_num = kernel_num
#         self.temperature = 1.0

#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
#         self.bn = nn.BatchNorm2d(attention_channel)
#         self.relu = nn.ReLU(inplace=True)

#         self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
#         self.func_channel = self.get_channel_attention

#         if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
#             self.func_filter = self.skip
#         else:
#             self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
#             self.func_filter = self.get_filter_attention

#         if kernel_size == 1:  # point-wise convolution
#             self.func_spatial = self.skip
#         else:
#             self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
#             self.func_spatial = self.get_spatial_attention

#         if kernel_num == 1:
#             self.func_kernel = self.skip
#         else:
#             self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
#             self.func_kernel = self.get_kernel_attention

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def update_temperature(self, temperature):
#         self.temperature = temperature

#     @staticmethod
#     def skip(_):
#         return 1.0

#     def get_channel_attention(self, x):
#         channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
#         return channel_attention

#     def get_filter_attention(self, x):
#         filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
#         return filter_attention

#     def get_spatial_attention(self, x):
#         spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
#         spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
#         return spatial_attention

#     def get_kernel_attention(self, x):
#         kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
#         kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
#         return kernel_attention

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = self.fc(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
# class ODConv2d(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#                  reduction=0.0625, kernel_num=4):
#         super(ODConv2d, self).__init__()
#         self.in_planes = in_planes
#         self.out_planes = out_planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.kernel_num = kernel_num
#         self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
#                                    reduction=reduction, kernel_num=kernel_num)
#         self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
#                                    requires_grad=True)
#         self._initialize_weights()

#         if self.kernel_size == 1 and self.kernel_num == 1:
#             self._forward_impl = self._forward_impl_pw1x
#         else:
#             self._forward_impl = self._forward_impl_common

#     def _initialize_weights(self):
#         for i in range(self.kernel_num):
#             nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

#     def update_temperature(self, temperature):
#         self.attention.update_temperature(temperature)

#     def _forward_impl_common(self, x):
#         # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
#         # while we observe that when using the latter method the models will run faster with less gpu memory cost.
#         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
#         batch_size, in_planes, height, width = x.size()
#         x = x * channel_attention
#         x = x.reshape(1, -1, height, width)
#         aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
#         aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
#             [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
#         output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
#                           dilation=self.dilation, groups=self.groups * batch_size)
#         output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
#         output = output * filter_attention
#         return output

#     def _forward_impl_pw1x(self, x):
#         channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
#         x = x * channel_attention
#         output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
#                           dilation=self.dilation, groups=self.groups)
#         output = output * filter_attention
#         return output

#     def forward(self, x):
#         return self._forward_impl(x)
# class DWConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
#         """
#         Depthwise Separable Convolution
#         Consists of:
#         1. Depthwise Convolution: Applies a single convolutional filter per input channel.
#         2. Pointwise Convolution: Applies 1x1 convolutions to combine the output of the depthwise layer.

#         Args:
#             in_channels (int): Number of input channels.
#             out_channels (int): Number of output channels.
#             kernel_size (int): Size of the convolution kernel. Default is 3.
#             stride (int): Stride of the convolution. Default is 1.
#             padding (int): Zero-padding added to both sides of the input. Default is 1.
#             bias (bool): If True, adds a learnable bias to the output. Default is False.
#         """
#         super(DWConv, self).__init__()
#         # Depthwise Convolution
#         self.depthwise = nn.Conv2d(
#             in_channels, in_channels, kernel_size=kernel_size,
#             stride=stride, padding=padding, groups=in_channels, bias=bias
#         )
#         # Pointwise Convolution
#         self.pointwise = nn.Conv2d(
#             in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
#         )

#     def forward(self, x):
#         x = self.depthwise(x)  # Apply depthwise convolution
#         x = self.pointwise(x)  # Apply pointwise convolution
#         return x
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = RepConv(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.caa = CAA(out_size)  # Apply CAA here

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        inputs2 = F.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)), mode='bilinear', align_corners=True)
        # inputs1 = self.caa(inputs1)
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
            # nn.Dropout(dropout_rate), 
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
