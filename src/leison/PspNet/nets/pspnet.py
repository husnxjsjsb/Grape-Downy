import torch
import torch.nn.functional as F
from torch import nn

from nets.mobilenetv2 import mobilenetv2
from nets.mobilenetv3 import mobilenet_v3
from nets.sim_mobilenetv3 import mobilenet_v31
from nets.resnet import resnet50
from efficientnet_pytorch import EfficientNet
# from nets.FasterNet import FasterNet
from nets.Starnet import starnet_s1
class FasterNetBackbone(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=False):
        super(FasterNetBackbone, self).__init__()
        from nets.FasterNet import FasterNet  # 确保正确导入 FasterNet
        
        # 初始化 FasterNet，设置 fork_feat=False 以输出单尺度特征
        self.model = FasterNet(
            fork_feat=False,
            depths=[1, 2, 8, 2],  # 根据任务调整深度
            embed_dim=30,          # 基础通道数
            pretrained=pretrained
        )
        self.downsample_factor = downsample_factor

    def forward(self, x):
        # FasterNet 的前向传播（输出单尺度特征）
        x = self.model.patch_embed(x)  # 初始 Patch Embedding
        x = self.model.stages(x)      # 通过所有阶段
        
        # 模拟其他 Backbone 的双分支输出（x_aux 和 x）
        # 注意：FasterNet 默认输出单尺度，这里需根据实际需求调整
        if self.downsample_factor == 8:
            x_aux = x  # 若无多尺度特征，暂用同一特征层
            x = x
        elif self.downsample_factor == 16:
            x_aux = x
            x = x
        else:
            raise ValueError('Unsupported downsample factor. Use 8 or 16.')
        
        return x_aux, x
class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        #--------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1[0]
        self.bn1 = model.conv1[1]
        self.relu1 = model.conv1[2]
        self.conv2 = model.conv1[3]
        self.bn2 = model.conv1[4]
        self.relu2 = model.conv1[5]
        self.conv3 = model.conv1[6]
        self.bn3 = model.bn1
        self.relu3 = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        #--------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x
class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV3, self).__init__()
        from functools import partial

        # 加载 MobileNetV3 模型
        model = mobilenet_v3(pretrained)
        self.features = model.features[:-1]  # 去掉最后的分类层

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]  # MobileNetV3 的下采样层索引

        # 根据下采样因子修改卷积的步长与膨胀系数
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

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
        # 提取低层特征（用于辅助分支）
        low_level_features = self.features[:4](x)
        # 提取高层特征（用于主干分支）
        x = self.features[4:](low_level_features)
        return low_level_features, x
class SIm_MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(SIm_MobileNetV3, self).__init__()
        from functools import partial

        # 加载 MobileNetV3 模型
        model = mobilenet_v31(pretrained)
        self.features = model.features[:-1]  # 去掉最后的分类层

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]  # MobileNetV3 的下采样层索引

        # 根据下采样因子修改卷积的步长与膨胀系数
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

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
        # 提取低层特征（用于辅助分支）
        low_level_features = self.features[:4](x)
        # 提取高层特征（用于主干分支）
        x = self.features[4:](low_level_features)
        return low_level_features, x
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
class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet50", pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        elif backbone == "fasternet":
            self.backbone = FasterNetBackbone(downsample_factor, pretrained)
            aux_channel = 36   # 根据 FasterNet 的实际通道数调整
            out_channel = 240  # 根据 FasterNet 的实际通道数调整
        elif backbone == "mobilenetv3":
            self.backbone = MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            aux_channel = 16  # 修改为 MobileNetV3 低层特征的通道数
            out_channel = 200  # 修改为 MobileNetV3 高层特征的通道数
        elif backbone == "sim_mobilenetv3":
            self.backbone = SIm_MobileNetV3(downsample_factor=downsample_factor, pretrained=pretrained)
            aux_channel = 16  # 修改为 MobileNetV3 低层特征的通道数
            out_channel = 200  # 修改为 MobileNetV3 高层特征的通道数
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 96
            out_channel = 320
        elif backbone == "efficientnetb0":
            self.backbone = EfficientNetBackbone(model_name='efficientnet-b0', pretrained=pretrained, downsample_factor=downsample_factor)
            aux_channel = 16
            out_channel = 112
        elif backbone == "starnet":
            self.backbone = StarNetBackbone(downsample_factor=downsample_factor, pretrained=pretrained)
            aux_channel = 48  # 根据 StarNet 的实际输出通道数进行调整
            out_channel = 192  # 根据 StarNet 的实际输出通道数进行调整
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50, fasternet, starnet.'.format(backbone))

        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
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
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        #-----------------------------------------------------#
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])
        
        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


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