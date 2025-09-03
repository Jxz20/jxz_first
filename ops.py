from __future__ import absolute_import, division

import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

#权值初始化。 基本都是经验值 比如凯明初始化 Xavier初始化之类的


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)   # Xavier均匀分布
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 常数值初始化
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb'] #ltwh是左上角和宽高   ltrb是左上角和右下角
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness) #画矩形框  需要左上角和右下角的坐标
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img


#对图像裁剪


def crop_and_resize(img, center, size, out_size,  # 裁剪中心 裁剪大小 最后输出大小
                    border_type=cv2.BORDER_CONSTANT,  # 边缘填充方式是固定值填充
                    border_value=(0, 0, 0),  # 默认固定值填充；本文是通道均值填充
                    interp=cv2.INTER_LINEAR):  # resize的方法是线性插值法
    # convert box to corners (0-indexed)
    size = round(size)  # round 是对数组四舍五入
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))  # corners 是左上和右下的坐标
    corners = np.round(corners).astype(int)  # 取整

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))  # 判断要裁剪的size有无超出image的边界
    if npad > 0:  # 如果超出则先对图像做填充
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)  # corners相对坐标转化
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]#图像剪裁

    #corners[0]:corners[2]：这部分表示子图像在垂直方向上的范围，从 corners 数组中取得左上角的 y 坐标到右下角的 y 坐标。
    #corners[1]:corners[3]：这部分表示子图像在水平方向上的范围，从 corners 数组中取得左上角的 x 坐标到右下角的 x 坐标。
    #因此，patch 将包含从 img 中指定范围（由 corners 定义）内的图像数据。这通常用于从大图像中提取特定区域或目标区域的小图像。

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)#对图像进行缩放


    # 显示图像
    plt.imshow(patch)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    return patch

