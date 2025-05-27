import random

import numpy as np
import torch
from PIL import Image


#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'vgg'              : 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19'            : 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'resnet50'         : 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
        'swin_transformer' : 'https://objects.githubusercontent.com/github-production-release-asset-2e65be/357198522/fd006b80-9bd3-11eb-8445-769d89efab4e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20240828%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240828T113951Z&X-Amz-Expires=300&X-Amz-Signature=9ee6e8090fe21795b83c2cc046b602ba65894ba4bc934797d5d8fd060e3e40d4&X-Amz-SignedHeaders=host&actor_id=102045635&key_id=0&repo_id=357198522&response-content-disposition=attachment%3B%20filename%3Dswin_tiny_patch4_window7_224.pth&response-content-type=application%2Foctet-stream',
        'efficientnet_b2'  : 'https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth',
    }
    
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
