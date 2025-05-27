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
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
from nets.unet import Unet as unet

from utils.utils import cvtColor, preprocess_input, resize_image, show_config
class Unet1(object):
    _defaults = {
        "model_path": r"C:\model\unet-pytorch-main\train_model\Unet主干\resnet.pth",
        "num_classes": 3,
        "backbone": "resnet50",
        "input_shape": [512, 512],
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

    # def generate(self, onnx=False):
    #     self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     self.net.load_state_dict(torch.load(self.model_path, map_location=device))
    #     self.net = self.net.eval()
    #     if not onnx:
    #         if self.cuda:
    #             self.net = nn.DataParallel(self.net)
    #             self.net = self.net.cuda()
    def generate(self, onnx=False):
     self.net = unet(num_classes=self.num_classes, backbone=self.backbone,)
    
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

    # 计算模型的参数量
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.net.parameters())
        total_params_in_million = total_params / 1e6  # 转换为百万（M）
        return total_params_in_million

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
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image


if __name__ == "__main__":
    miou_mode = 0
    num_classes = 3
    name_classes = ["_background_", "disease","leaf"]
    VOCdevkit_path = r'C:\\model\\unet-pytorch-main\\unet-pytorch-main\\VOCdevkit'

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
        input_tensor = torch.randn(1, 3, 512, 512).cuda()
        total_params = unet.count_parameters()
        inference_time = unet.measure_inference_time(input_tensor)
        fps = 1000 / inference_time
        gflops = unet.calculate_gflops()
        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/simam/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, _ = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, [0]*num_classes, name_classes)

        # 计算各项指标
        pa_background = round(PA_Recall[0] * 100, 2)
        pa_disease = round(PA_Recall[1] * 100, 2)
        mpa_bg_dis = round((pa_background + pa_disease) / 2, 2)
        iou_background = round(IoUs[0] * 100, 2)
        iou_disease = round(IoUs[1] * 100, 2)
        miou_bg_dis = round((iou_background + iou_disease) / 2, 2)
        fps = round(fps, 2)
        gflops = round(gflops, 2)
        total_params = round(total_params, 2)
        inference_time = round(inference_time, 2)

        csv_file_path = os.path.join(miou_out_path, 'test_results.csv')
        header = ["Model", "PA(background)", "PA(disease)", "mPA(background_disease)", 
                  "IoU(background)", "IoU(disease)", "mIoU(background_disease)", 
                  "FPS", "GFLOPs", "Params (M)", "Inference Time (ms)"]

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "segformer",
                pa_background,
                pa_disease,
                mpa_bg_dis,
                iou_background,
                iou_disease,
                miou_bg_dis,
                fps,
                gflops,
                total_params,
                inference_time
            ])

        print(f"Results saved to {csv_file_path}")
