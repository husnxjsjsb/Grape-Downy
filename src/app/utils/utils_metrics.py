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

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

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

def save_results_to_csv(miou_out_path, IoUs, PA_Recall, Accuracy):
    
    model_name = 'efficientnet-Unet'
    
    #------------------------------------------------#
    #   准备CSV文件的头和内容
    #------------------------------------------------#
    header = ['model', 'mpa','iou(background)', 'iou(leaf)',    'miou', 'accuracy']
    # `IoUs`包含每类的 IoU，假设第0类是背景，第1类是叶子，第2类是病害
    data = [
        model_name, 
        np.nanmean(PA_Recall) * 100,  # mPA
        IoUs[0] * 100,                # IoU for background
        IoUs[1] * 100,                # IoU for leaf
        np.nanmean(IoUs) * 100,       # mIoU
        Accuracy * 100                # Accuracy
    ]
    
    #------------------------------------------------#
    #   保存为CSV文件，并确保新数据追加到文件末尾
    #------------------------------------------------#
    csv_file_path = os.path.join(miou_out_path, "model_results.csv")
    
    # Check if the file exists, if not, write the header first
    file_exists = os.path.exists(csv_file_path)
    
    with open(csv_file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(header)
        
        # Append the new data
        writer.writerow(data)

# 调用这个新函数来生成CSV
def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    save_results_to_csv(miou_out_path, IoUs, PA_Recall, per_Accuracy(hist))


            