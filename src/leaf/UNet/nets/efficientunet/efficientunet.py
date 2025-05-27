import torch
import torch.nn as nn
from collections import OrderedDict
from .layers import *
from .efficientnet import EfficientNet


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        inputs2 = self.up(inputs2)
        if inputs2.size(2) != inputs1.size(2) or inputs2.size(3) != inputs1.size(3):
            inputs2 = nn.functional.interpolate(inputs2, size=(inputs1.size(2), inputs1.size(3)), mode='bilinear', align_corners=True)
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super(EfficientUnet, self).__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.in_filters = self.get_in_filters()  # 使用新的方法获取输入通道数
        self.out_filters = [64, 128, 256, 512]

        self.up_concat4 = UnetUp(self.in_filters[3] + self.in_filters[4], self.out_filters[3])  # 80 + 160
        self.up_concat3 = UnetUp(self.in_filters[2] + self.out_filters[3], self.out_filters[2])  # 40 + 512
        self.up_concat2 = UnetUp(self.in_filters[1] + self.out_filters[2], self.out_filters[1])  # 24 + 256
        self.up_concat1 = UnetUp(self.in_filters[0] + self.out_filters[1], self.out_filters[0])  # 16 + 128

        self.final_conv = nn.Conv2d(self.out_filters[0], out_channels, kernel_size=1)

    def get_in_filters(self):
        n_channels_dict = {
            'efficientnet-b0': 1280, 'efficientnet-b1': 1280,
            'efficientnet-b2': 1408, 'efficientnet-b3': 1536,
            'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
            'efficientnet-b6': 2304, 'efficientnet-b7': 2560
        }
        return [16, 24, 40, 80, 160]  # 从backbone获取的特征图通道数

    def forward(self, x):
        input_ = x
        blocks = self.get_blocks_to_be_concat(x)
        _, x = blocks.popitem()

        up4 = self.up_concat4(blocks.popitem()[1], x)
        up3 = self.up_concat3(blocks.popitem()[1], up4)
        up2 = self.up_concat2(blocks.popitem()[1], up3)
        up1 = self.up_concat1(blocks.popitem()[1], up2)

        final = self.final_conv(up1)
        return final

    def get_blocks_to_be_concat(self, x):
        shapes = set()
        blocks = OrderedDict()
        hooks = []
        count = 0

        def register_hook(module):
            def hook(module, input, output):
                nonlocal count
                try:
                    if module.name == f'blocks_{count}_output_batch_norm':
                        count += 1
                        shape = output.size()[-2:]
                        if shape not in shapes:
                            shapes.add(shape)
                            blocks[module.name] = output
                    elif module.name == 'head_swish':
                        blocks.popitem()
                        blocks[module.name] = output
                except AttributeError:
                    pass

        self.encoder.apply(register_hook)
        self.encoder(x)

        for h in hooks:
            h.remove()

        return blocks


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder('efficientnet-b0', pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model

# 其他 get 函数保持不变...
