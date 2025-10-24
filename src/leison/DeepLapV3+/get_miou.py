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
        "model_path": r"",
        "num_classes": 3,
        "backbone": "",#主干网络
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
     self.net = unet(num_classes=self.num_classes, downsample_factor=self.downsample_factor, pretrained=False, backbone=self.backbone)
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 明确设备选择逻辑
     device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')
    
    # 先加载到CPU，避免设备映射混乱
     state_dict = torch.load(self.model_path, map_location='cpu')  
     self.net.load_state_dict(state_dict)
    
    # 迁移模型到目标设备
     self.net = self.net.to(device)
    
    # DataParallel包装（仅在多GPU时启用）
     if self.cuda and torch.cuda.device_count() > 1:
         self.net = nn.DataParallel(self.net)
    
     self.net = self.net.eval()
     print(f'Model loaded on {device}')

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.net.parameters())
        return total_params / 1e6

    def measure_inference_time(self, input_tensor, num_runs=100):
        self.net.eval()
        total_time = 0.0
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.net(input_tensor)
                total_time += time.time() - start_time
        return (total_time / num_runs) * 1000

    def calculate_gflops(self, input_shape=(1, 3, 512, 512)):
        from thop import profile
        model = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
        input = torch.randn(*input_shape)
        if self.cuda:
            input = input.cuda()
        flops, _ = profile(model, inputs=(input,), verbose=False)
        return flops / 1e9

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
    num_classes = 3
    # 类别顺序: [0: _background_, 1: disease, 2: leaf]
    name_classes = ["_background_", "disease", "leaf"] 
    VOCdevkit_path = r'data/VOCdevkit_disease'

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
        
        # 确保输入张量在正确设备上
        device = next(unet.net.parameters()).device
        input_tensor = torch.randn(1, 3, 512, 512).to(device)
        
        total_params = unet.count_parameters()
        inference_time = unet.measure_inference_time(input_tensor)
        fps = 1000 / inference_time
        
        
        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/simam/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        # IoUs[0] = background, IoUs[1] = disease, IoUs[2] = leaf
        hist, IoUs, PA_Recall, _ = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        
        # 保持原始 show_results 调用（它会输出所有 3 个类别的结果到文本文件）
        show_results(miou_out_path, hist, IoUs, PA_Recall, [0]*num_classes, name_classes)

        # --- 新的二分类指标计算逻辑 (disease vs non-disease) ---
        
        # 1. Non-Disease (Background + Leaf) IoU
        # 取 background (0) 和 leaf (2) 的 IoU 平均值
        iou_non_disease = (IoUs[0] + IoUs[2]) / 2.0
        pa_non_disease = (PA_Recall[0] + PA_Recall[2]) / 2.0

        # 2. Disease IoU (类别 1)
        iou_disease = IoUs[1]
        pa_disease = PA_Recall[1]
        
        # 3. Binary mIoU (只计算 disease 和 non-disease 两个类别的平均)
        miou_binary = (iou_disease + iou_non_disease) / 2.0
        mpa_binary = (pa_disease + pa_non_disease) / 2.0


        # 舍入结果并转换为百分比
        pa_non_disease = round(pa_non_disease * 100, 2)
        pa_disease = round(pa_disease * 100, 2)
        mpa_binary = round(mpa_binary * 100, 2)
        
        iou_non_disease = round(iou_non_disease * 100, 2)
        iou_disease = round(iou_disease * 100, 2)
        miou_binary = round(miou_binary * 100, 2)
        
        # 性能指标
        fps = round(fps, 2)
        
        total_params = round(total_params, 2)
        inference_time = round(inference_time, 2)

        # --- CSV 写入部分 ---
        # 使用一个新的文件名，以区分是二分类聚合的结果
        csv_file_path = os.path.join(miou_out_path, 'evaluation_results_binary_aggregated.csv')
        
        # 更新表头，只保留 disease 和 aggregated background 的相关指标
        header = [
            "MobilenetV3", 
            "PA(Non-Disease: BG+Leaf)", 
            "PA(Disease)", 
            "mPA(Binary)",
            "IoU(Non-Disease: BG+Leaf)", 
            "IoU(Disease)", 
            "mIoU(Binary)", 
            "FPS", 
            "GFLOPs", 
            "Params (M)", 
            "Inference Time (ms)"
        ]

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "VGG",
                pa_non_disease,
                pa_disease,
                mpa_binary,
                iou_non_disease,
                iou_disease,
                miou_binary,
                fps,
                
                total_params,
                inference_time
            ])

        print(f"Binary aggregated results saved to {csv_file_path}")
        print("\n--- 二分类聚合 IoU/mIoU 结果 ---")
        print(f"IoU(Disease): {iou_disease:.2f}%")
        print(f"IoU(Non-Disease: BG+Leaf): {iou_non_disease:.2f}%")
        print(f"mIoU(Binary): {miou_binary:.2f}%")