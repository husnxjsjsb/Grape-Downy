import torch 
import torch.nn  as nn 
import math 
 
 
__all__ = ['mobilenetv4_conv_small', 'mobilenetv4_conv_medium', 'mobilenetv4_conv_large',
           'mobilenetv4_hybrid_medium', 'mobilenetv4_hybrid_large']
 
 
def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor 
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor 
    return new_value 
 
 
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBN, self).__init__()
        self.block  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x) 
 
 
class UniversalInvertedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 stride,
                 middle_dw_downsample: bool = True,
                 use_layer_scale: bool = False,
                 layer_scale_init_value: float = 1e-5):
        super(UniversalInvertedBottleneck, self).__init__()
        self.start_dw_kernel_size  = start_dw_kernel_size 
        self.middle_dw_kernel_size  = middle_dw_kernel_size 
 
        if start_dw_kernel_size:
           self.start_dw_conv  = nn.Conv2d(in_channels, in_channels, start_dw_kernel_size, 
                                          stride if not middle_dw_downsample else 1,
                                          (start_dw_kernel_size - 1) // 2,
                                          groups=in_channels, bias=False)
           self.start_dw_norm  = nn.BatchNorm2d(in_channels)
        
        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv  = nn.Conv2d(in_channels, expand_channels, 1, 1, bias=False)
        self.expand_norm  = nn.BatchNorm2d(expand_channels)
        self.expand_act  = nn.ReLU(inplace=True)
 
        if middle_dw_kernel_size:
           self.middle_dw_conv  = nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size,
                                           stride if middle_dw_downsample else 1,
                                           (middle_dw_kernel_size - 1) // 2,
                                           groups=expand_channels, bias=False)
           self.middle_dw_norm  = nn.BatchNorm2d(expand_channels)
           self.middle_dw_act  = nn.ReLU(inplace=True)
        
        self.proj_conv  = nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False)
        self.proj_norm  = nn.BatchNorm2d(out_channels)
 
        if use_layer_scale:
            self.gamma  = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),  requires_grad=True)
 
        self.use_layer_scale  = use_layer_scale 
        self.identity  = stride == 1 and in_channels == out_channels 
 
    def forward(self, x):
        shortcut = x 
 
        if self.start_dw_kernel_size: 
            x = self.start_dw_conv(x) 
            x = self.start_dw_norm(x) 
 
        x = self.expand_conv(x) 
        x = self.expand_norm(x) 
        x = self.expand_act(x) 
 
        if self.middle_dw_kernel_size: 
            x = self.middle_dw_conv(x) 
            x = self.middle_dw_norm(x) 
            x = self.middle_dw_act(x) 
 
        x = self.proj_conv(x) 
        x = self.proj_norm(x) 
 
        if self.use_layer_scale: 
            x = self.gamma  * x 
 
        return x + shortcut if self.identity  else x 
 
 
class MobileNetV4(nn.Module):
    def __init__(self, block_specs, num_classes=1000):
        super(MobileNetV4, self).__init__()
 
        c = 3 
        layers = []
        for block_type, *block_cfg in block_specs:
            if block_type == 'conv_bn':
                block = ConvBN 
                k, s, f = block_cfg 
                layers.append(block(c,   f, k, s))
            elif block_type == 'uib':
                block = UniversalInvertedBottleneck 
                start_k, middle_k, s, f, e = block_cfg 
                layers.append(block(c,   f, e, start_k, middle_k, s))
            elif block_type == 'mhsa':
                block = MHSA 
                layers.append(block(c))  
            else:
                raise NotImplementedError 
            c = f 
        self.features   = nn.Sequential(*layers)
        # building last several layers 
        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))
        hidden_channels = 256  # 调整为较小的数值 
        self.conv   = ConvBN(c, hidden_channels, 1)
        self.classifier   = nn.Linear(hidden_channels, num_classes)
 
    def forward(self, x):
        # 提取低层特征（例如第 4 层的输出）
        low_level_features = None 
        for i, layer in enumerate(self.features):  
            x = layer(x)
            if i == 3:  # 假设第 4 层是低层特征 
                low_level_features = x 
 
        # 提取高层特征 
        high_level_features = x  # 在应用 avgpool 之前保存高层特征 
        x = self.avgpool(x)  
        x = self.conv(x)  
        x = x.view(x.size(0),   -1)
        x = self.classifier(x)  
 
        return low_level_features, high_level_features  # 返回两个4D张量 
 
def mobilenetv4_conv_small(**kwargs):
    """
    Constructs a MobileNetV4-Conv-Small model
    """
    block_specs = [
        # conv_bn, kernel_size, stride, out_channels
        # uib, start_dw_kernel_size, middle_dw_kernel_size, stride, out_channels, expand_ratio
        # 112px
        ('conv_bn', 3, 2, 32),
        # 56px
        ('conv_bn', 3, 2, 32),
        ('conv_bn', 1, 1, 32),
        # 28px
        ('conv_bn', 3, 2, 24),  # 修改 96 -> 24
        ('conv_bn', 1, 1, 24),  # 修改 64 -> 24
        # 14px
        ('uib', 5, 5, 2, 24, 3.0),  # 修改 96 -> 24
        ('uib', 0, 3, 1, 24, 2.0),  # 修改 96 -> 24
        ('uib', 0, 3, 1, 24, 2.0),  # 修改 96 -> 24
        ('uib', 0, 3, 1, 24, 2.0),  # 修改 96 -> 24
        ('uib', 0, 3, 1, 24, 2.0),  # 修改 96 -> 24
        ('uib', 3, 0, 1, 24, 4.0),  # 修改 96 -> 24
        # 7px
        ('uib', 3, 3, 2, 128, 6.0),  # 保持不变
        ('uib', 5, 5, 1, 128, 4.0),  # 保持不变
        ('uib', 0, 5, 1, 128, 4.0),  # 保持不变
        ('uib', 0, 5, 1, 128, 3.0),  # 保持不变
        ('uib', 0, 3, 1, 128, 4.0),  # 保持不变
        ('uib', 0, 3, 1, 128, 4.0),  # 保持不变
        ('conv_bn', 1, 1, 320),  # 保持不变
    ]
    return MobileNetV4(block_specs, **kwargs)
