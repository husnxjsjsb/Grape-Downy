import colorsys
import io
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from torch import nn
from utils.utils import cvtColor, preprocess_input, resize_image
from nets.deeplabv3_plus import DeepLab as deep
import colorsys
import csv
import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch import nn
from nets.unet_efficientnet import Unet as unet
from nets.deeplabv3_plus import DeepLab as deep
# from nets.pspnet import PSPNet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
# ---------------------------- 模型定义部分 ----------------------------
class Unet1:
    _defaults = {
        "model_path": r"model\uneteff_sam_rep.pth",
        "num_classes": 3,
        "backbone": "efficientnetb0",
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

    def generate(self):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location='cpu')
        self.net.load_state_dict(state_dict)
        self.net = self.net.to(device)
        if self.cuda and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net = self.net.eval()

    def detect_single_image(self, image):
        original_shape = np.array(image).shape[:2]
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        image_tensor = torch.from_numpy(image_data)
        if self.cuda:
            image_tensor = image_tensor.cuda()
        with torch.no_grad():
            pr = self.net(image_tensor)
            pr = F.softmax(pr.permute(0, 2, 3, 1), dim=-1).cpu().numpy()[0]
        pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
             int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        return pr.argmax(axis=-1)

class DeeplabV3(nn.Module):
    _defaults = {
        "model_path": r"model\sim_mobilenetv3.pth",
        "num_classes": 2,
        "backbone": "mv3",
        "input_shape": [512, 512],
        "downsample_factor": 8,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(self._defaults)
        self.colors = [(0, 0, 0), (0, 128, 0)]
        self.generate()

    def generate(self):
        self.net = deep(num_classes=self.num_classes, backbone=self.backbone,
                       downsample_factor=self.downsample_factor, pretrained=False)
        device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.to(device)
        if self.cuda and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net = self.net.eval()

    def detect_image(self, image):
        image = cvtColor(image)
        original_h, original_w = image.size[1], image.size[0]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()
        with torch.no_grad():
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            return pr.argmax(axis=-1)

# ---------------------------- FastAPI 服务部分 ----------------------------
app = FastAPI()

# 初始化模型
@app.on_event("startup")
async def load_models():
    global DEEP, UNET
    DEEP = DeeplabV3()
    UNET = Unet1()
    print("Models loaded successfully")

def create_overlay(original, mask):
    """创建带轮廓的叠加图像"""
    overlay = Image.new('RGBA', original.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # 处理病害区域（红色）
    disease_mask = (mask == 1).astype(np.uint8)
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [(p[0][0], p[0][1]) for p in approx]
        if len(points) >= 3:
            draw.polygon(points, outline=(255, 0, 0, 200), width=3)
    
    # 处理健康区域（绿色）
    healthy_mask = (mask == 2).astype(np.uint8)
    contours, _ = cv2.findContours(healthy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = [(p[0][0], p[0][1]) for p in approx]
        if len(points) >= 3:
            draw.polygon(points, outline=(0, 255, 0, 200), width=3)
    
    return Image.alpha_composite(original.convert('RGBA'), overlay).convert('RGB')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 读取上传图片
        image_data = await file.read()
        original_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 第一步：Deeplab处理
        start_time = time.time()
        deeplab_mask = DEEP.detect_image(original_image)
        
        # 第二步：生成组合图像（内存操作）
        combined = Image.composite(
            original_image,
            Image.new('RGB', original_image.size, (0, 0, 0)),
            Image.fromarray((deeplab_mask > 0).astype(np.uint8) * 255)
        )
        
        # 第三步：UNet处理
        unet_result = UNET.detect_single_image(combined)
        inference_time = time.time() - start_time
        
        # 计算病害比例
        disease_pixels = np.sum(unet_result == 1)
        healthy_pixels = np.sum(unet_result == 2)
        total = disease_pixels + healthy_pixels
        severity = disease_pixels / total if total > 0 else 0
        
        # 生成可视化结果
        overlay_image = create_overlay(original_image, unet_result)
        
        # 转换为字节流
        img_byte_arr = io.BytesIO()
        overlay_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        overlay_image = create_overlay(original_image, unet_result)

# 转换为字节流
        output_buffer = io.BytesIO()
        overlay_image.save(output_buffer, format='JPEG')
        output_buffer.seek(0)

        
        # 修改前（错误）：
        severity_percentage = f"{severity * 100:.2f}%"

# 修改后（正确）：
        severity_value = severity   # 保留原始数值
        return JSONResponse(content={
    "image_name": file.filename,
    "severity_percentage": float(f"{severity_value:.2f}"),  # 显式转换为浮点
    "overlay_image": output_buffer.getvalue().hex()
})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="172.29.16.6", port=8000)