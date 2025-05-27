import colorsys
import csv
import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional  as F
from PIL import Image
from tqdm import tqdm
from concurrent.futures  import ThreadPoolExecutor
from torch import nn
# from nets.nets.pspnet  import PSPNet as psp
from utils.utils  import cvtColor, preprocess_input, resize_image
from nets.unet_efficientnet  import Unet as unet
from nets.deeplabv3_plus import DeepLab as deep # 假设这是你定义的 Deeplabv3+ 模型
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

# 假设这是你定义的 Deeplabv3+ 模型
from nets.unet_efficientnet import Unet as unet
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
# from nets.unet_efficientnet import Unet as unet
from nets.deeplabv3_plus import DeepLab as deep

from nets.unet_vgg16 import Unet as unet
class Unet1(object):
    _defaults = {
        "model_path": r"C:\model\unet-pytorch-main\train_model\UNet对比\UnetVGG.pth",
        "num_classes": 3,
        "backbone": "vgg",
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

    def generate(self, onnx=False):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_single_image(self, image, count=False, name_classes=None):
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
        pr_class = pr.argmax(axis=-1)
        pr_confidence = pr.max(axis=-1)

        average_confidence = []
        if count:
            classes_nums = np.zeros(self.num_classes)
            total_points = original_shape[0] * original_shape[1]

            for j in range(self.num_classes):
                mask = (pr_class == j)
                classes_nums[j] = np.sum(mask)
                if classes_nums[j] > 0:
                    average_confidence.append(np.mean(pr_confidence[mask]))
                else:
                    average_confidence.append(0.0)

            return pr_class, classes_nums, average_confidence
        return pr_class


class DeeplabV3(nn.Module):
    _defaults = {
        "model_path":"C:\model\deeplabv3-plus-pytorch-main2\deep\主干\deepv2.pth",
        "num_classes": 2,
        "backbone": "mobilenetv2",
        "input_shape": [512, 512],
        "downsample_factor": 8,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        super(DeeplabV3, self).__init__()
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.colors = [(0, 0, 0), (0, 128, 0)]
        self.generate()

    def generate(self):
        self.net = deep(num_classes=self.num_classes, backbone=self.backbone,
                        downsample_factor=self.downsample_factor, pretrained=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()

        if self.cuda and torch.cuda.is_available():
            self.net = nn.DataParallel(self.net).cuda()

    def detect_image(self, image):
        image = cvtColor(image)
        orininal_h, orininal_w = image.size[1], image.size[0]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        images = torch.from_numpy(image_data)

        if self.cuda and torch.cuda.is_available():
            images = images.cuda()

        with torch.no_grad():
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr_class = pr.argmax(axis=-1)

        seg_img = np.uint8(pr_class)
        image = Image.fromarray(seg_img, mode='P')

        palette = []
        for color in self.colors:
            palette.extend(color)
        image.putpalette(palette)

        return image


def process_image_and_mask_torch(image_path, mask_path, output_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        return

    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_tensor = torch.from_numpy(image).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)

    binary_mask = (mask_tensor > 0).float()
    masked_image = image_tensor * binary_mask.unsqueeze(2)
    result_image = masked_image.cpu().numpy().astype(np.uint8)

    cv2.imwrite(output_path, result_image)


def deleteIntermediateDirectories(directory_path, keep_first, keep_second):
    for dir_name in os.listdir(directory_path):
        if dir_name in [keep_first, keep_second]:
            continue

        target_path = os.path.join(directory_path, dir_name)
        try:
            if os.path.isdir(target_path):
                for root, dirs, files in os.walk(target_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(target_path)
                print(f"成功删除中间目录: {target_path}")
            else:
                os.remove(target_path)
        except Exception as e:
            print(f"删除失败 {target_path}: {str(e)}")


if __name__ == "__main__":
    ROOT_DIR = r"C:\model\segformer-pytorch-master\VOCdevkit\VOC2007\JPEGImages"
    val_txt_path = r"C:\model\unet-pytorch-main\unet-pytorch-main\VOCdevkit\VOC2007\ImageSets\Segmentation\val.txt"
    output_dir = r"C:\download\data\hebing"
    save_dir_mask = r"C:\download\data\unetpng"

    with open(val_txt_path, "r") as f:
        base_names = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir_mask, exist_ok=True)

    DEEP = DeeplabV3()
    unet = Unet1()
    name_classes = ["_background_", "disease", "leaf"]

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    deep_params = count_parameters(DEEP.net)
    unet_params = count_parameters(unet.net)
    total_params = deep_params + unet_params

    csv_file_path = os.path.join(r"C:\model\unet-pytorch-main\unet-pytorch-main\miou_out", "inference_results.csv")
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        deep_params_m = deep_params / 1e6
        unet_params_m = unet_params / 1e6
        total_params_m = total_params / 1e6
        
        csv_writer.writerow(["DeeplabV3 Parameters (M)", "UNet Parameters (M)", 
                            "Total Parameters (M)", "Total Inference Time (s)"])
        
        total_inference_time = 0
        for base_name in tqdm(base_names, desc="Processing Images"):
            img_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                temp_path = os.path.join(ROOT_DIR, f"{base_name}{ext}")
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            if not img_path:
                print(f"Image {base_name}* not found, skipping.")
                continue

            img_name = os.path.basename(img_path)
            final_result_path = os.path.join(save_dir_mask, f"{base_name}.png")

            if os.path.exists(final_result_path):
                continue

            try:
                start_time = time.time()

                image = Image.open(img_path)
                psp_mask = DEEP.detect_image(image)
                mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                psp_mask.save(mask_path)

                combined_path = os.path.join(output_dir, f"{base_name}_combined.jpg")
                process_image_and_mask_torch(img_path, mask_path, combined_path)

                combined_image = Image.open(combined_path)
                unet_result, _, _ = unet.detect_single_image(combined_image, count=True, name_classes=name_classes)

                seg_img = Image.fromarray(np.uint8(unet_result), mode='P')
                seg_img.putpalette([val for color in unet.colors for val in color])
                seg_img.save(final_result_path)

                end_time = time.time()
                total_inference_time += end_time - start_time

            except Exception as e:
                print(f"Error processing {base_name}: {str(e)}")

        csv_writer.writerow([
            f"{deep_params_m:.2f}",
            f"{unet_params_m:.2f}",
            f"{total_params_m:.2f}",
            f"{total_inference_time:.2f}"
        ])

    deleteIntermediateDirectories(
        directory_path=r"C:\download\data",
        keep_first=save_dir_mask,
        keep_second=ROOT_DIR
    )