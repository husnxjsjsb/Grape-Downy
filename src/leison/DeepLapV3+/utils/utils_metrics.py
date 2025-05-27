import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def per_class_PA_Recall(hist):
    return np.diag(hist) / hist.sum(1)

def per_class_Precision(hist):
    return np.diag(hist) / hist.sum(0)

def per_Accuracy(hist):
    return np.diag(hist).sum() / hist.sum()

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):  
    print('Num classes:', num_classes)  
    # Initialize a global confusion matrix for all images
    hist = np.zeros((num_classes, num_classes))
    
    # Get paths for ground truth and prediction images
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]  

    # Process each image
    for ind in range(len(gt_imgs)): 
        # Read the prediction and label images
        pred = np.array(Image.open(pred_imgs[ind]))  
        label = np.array(Image.open(gt_imgs[ind]))  

        # Skip if the image and label dimensions don't match
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # Compute hist for individual image to get its IoU
        hist_single = fast_hist(label.flatten(), pred.flatten(), num_classes)
        IoU_single = per_class_iu(hist_single)

        # Print per-class IoU for this image
        if name_classes is not None:
            print(f'Image {png_name_list[ind]}:')
            for i, class_name in enumerate(name_classes):
                print(f'  {class_name} IoU: {round(IoU_single[i] * 100, 2)}%')
        
        # Accumulate to the global hist
        hist += hist_single

        # Output running mIoU, mPA, and Accuracy every 10 images
        if ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIoU-{:.2f}%; mPA-{:.2f}%; Accuracy-{:.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )

    # Compute metrics on the entire dataset
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    # Output the average IoU per class across all images
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIoU-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    # Print the final metrics
    print('===> mIoU: {:.2f}%; mPA: {:.2f}%; Accuracy: {:.2f}%'.format(
        np.nanmean(IoUs) * 100, 
        np.nanmean(PA_Recall) * 100, 
        per_Accuracy(hist) * 100
    ))

    return np.array(hist, np.int_), IoUs, PA_Recall, Precision
def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12, model_name='Model'):
    # 计算背景和叶子的IoU
    background_iou = IoUs[0]  # 假设背景是第一个类别
    leaf_iou = IoUs[1]  # 假设叶子是第二个类别

    # 计算mIoU
    mIoU = np.nanmean(IoUs) * 100

    # 计算mPA
    mPA = np.nanmean(PA_Recall) * 100

    # 计算Accuracy
    accuracy = per_Accuracy(hist) * 100

    # 保存指标到CSV文件
    metrics = [model_name, mPA, background_iou * 100, leaf_iou * 100, mIoU, accuracy]
    
    # 确保输出路径存在
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    # 读取现有CSV文件并追加新数据
    csv_file = os.path.join(miou_out_path, r"C:\model\deeplabv3-plus-pytorch-main2\miou_out\val_result.csv")
    header = ['model', 'mpa', 'iou(background)', 'iou(leaf)', 'miou', 'accuracy']

    # 如果文件不存在，则写入表头和数据
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(metrics)
        print(f"Metrics saved to {csv_file}")
    else:
        # 如果文件存在，则只追加数据
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics)
        print(f"Metrics appended to {csv_file}")

    # 继续生成图表
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(mIoU), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(mPA), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    # 保存混淆矩阵
    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            