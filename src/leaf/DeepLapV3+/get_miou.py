import os
import csv
from PIL import Image
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils.utils_metrics import compute_mIoU, show_results
import colorsys
import copy
import json
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.deeplabv3_plus import DeepLab as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
from thop import profile  

class Unet1(object):
    _defaults = {
        "model_path": r"",#模型路径
        "num_classes": 2,
        "backbone": "sim_mobilenetv3",
        "input_shape": [512, 512],
        "downsample_factor": 8,
        "mix_type": 0,
        "cuda": True,
    }
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes <= 3:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        show_config(**self._defaults)

    def generate(self, onnx=False):
    # 载入模型与权值
     self.net = unet(num_classes=self.num_classes, downsample_factor=self.downsample_factor, pretrained=False, backbone=self.backbone)
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
     self.net = self.net.eval()
     print('{} model, and classes loaded.'.format(self.model_path))
     if not onnx:
        if self.cuda:
            self.net = self.net.cuda()  # 直接移动到 GPU，不使用 DataParallel
    # 计算模型的参数量
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.net.parameters())
        total_params_in_million = total_params / 1e6  # 转换为百万（M）
        return total_params_in_million

    # 计算推理时间
    def measure_inference_time(self, input_tensor, num_runs=100):
        self.net.eval()  # 切换到评估模式
        total_time = 0.0

        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.net(input_tensor)
                total_time += time.time() - start_time

        avg_time_ms = (total_time / num_runs) * 1000  # 平均推理时间 (ms)
        return avg_time_ms

    # 计算 FPS
    def calculate_fps(self, latency_ms):
        return 1000 / latency_ms

    # 计算 GFLOPs
    def calculate_gflops(self, input_tensor):
        self.net.eval()
        flops, _ = profile(self.net, inputs=(input_tensor,))
        gflops = flops / 1e9  # 转换为 GFLOPs
        return gflops

    def get_miou_png(self, image):
        # 确保 image 是 PIL.Image 对象
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 调整图像大小
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        # 将 PIL.Image 转换为 NumPy 数组并进行预处理
        image_data = np.array(image_data, np.float32)
        image_data = np.transpose(preprocess_input(image_data), (2, 0, 1))  # 预处理并调整通道顺序
        image_data = np.expand_dims(image_data, 0)  # 添加 batch 维度

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (image.size[0], image.size[1]), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image


if __name__ == "__main__":
    miou_mode = 0
    num_classes = 2
    name_classes = [ "_background_,","leaf"]
    VOCdevkit_path = r'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        unet = Unet1()
        print("Load model done.")

        # 打印参数量、GFLOPs 和推理时间
        input_tensor = torch.randn(1, 3, 512, 512).cuda()  # 根据你的输入尺寸修改
        total_params = unet.count_parameters()
        print(f"Total Parameters: {total_params:.2f} M")  # 显示单位为百万参数

        inference_time = unet.measure_inference_time(input_tensor)
        fps = unet.calculate_fps(inference_time)
        gflops = unet.calculate_gflops(input_tensor)

        print(f"Inference Time: {inference_time:.2f} ms")
        print(f"FPS: {fps:.2f}")
        print(f"GFLOPs: {gflops:.2f}")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  # 执行计算mIoU的函数
        PA = np.sum(np.diag(hist)) / np.sum(hist) * 100
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        # 保存结果到CSV文件
        csv_file_path = os.path.join(miou_out_path, 'test_results1.csv')

        # 检查文件是否存在，如果不存在则写入表头
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # 写入表头
                writer.writerow(["Model", "PA","mPA", *[f"IoU ({name})" for name in name_classes], "mIoU", "mP", "Params (M)", "Inference Time (ms)", "FPS", "GFLOPs"])

        # 追加新内容
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 写入数据
            writer.writerow([
                "mobilenetv3",
                PA,
                np.mean(PA_Recall) * 100,  # mPA
                *[IoUs[i] * 100 for i in range(len(IoUs))],  # 各类 IoU
                np.mean(IoUs) * 100,  # mIoU
                np.mean(Precision) * 100,  # Accuracy
                total_params,  # Params (M)
                inference_time,  # Inference Time (ms)
                fps,  # FPS
                gflops  # GFLOPs
            ])

        print(f"Results saved to {csv_file_path}")