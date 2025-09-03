#
#只有模板更新 做好模板更新优化
#mean_apce只取了前20帧的值 后期不用每帧都计算apce了

#

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
import pandas as pd
import matplotlib.pyplot as plt

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


__all__ = ['TrackerSiamFC']

box1 = []
flag = 2
f_count = 0

apce_list = []
Fmax_list = []
########
def apce(response):
    Fmax = np.max(response)  # 找到最大值
    Fmin = np.min(response)  # 找到最小值
    sum_val = 0
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            sum_val += (response[i, j] - Fmin) ** 2

    Fmean = sum_val / (response.shape[0] * response.shape[1])
    APCE = (Fmax - Fmin) ** 2 / Fmean

    return APCE, Fmax


sum_apce = 0
sum_Fmax = 0
counts = 1
mean_apce = 0
mean_Fmax = 0

# apce_max = 0


def update_model(response):
    global sum_apce
    global sum_Fmax
    global counts
    global mean_Fmax
    global mean_apce
    global f_count
    global apce_list
    global Fmax_list
    # global apce_max

    curr_apce, curr_Fmax = apce(response)
    # if counts < ((f_count // 100) - 1) or counts == 1:
    sum_apce += curr_apce
    sum_Fmax += curr_Fmax
    mean_apce = sum_apce / counts
    mean_Fmax = sum_Fmax / counts

    counts += 1

    # apce_list.append(curr_apce)
    # Fmax_list.append(curr_Fmax)

    beta1 = 1.1  # 这些系数的值好像不是非常理想 还得多试试？？
    beta2 = 0.9

    if curr_apce >= beta1 * mean_apce and curr_Fmax >= beta2 * mean_Fmax:
        curr_replace = 1
    else:
        curr_replace = 0

    return curr_replace, curr_apce, mean_apce



class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)

        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # 参数初始化
        # 模板更新
        global box1
        global flag
        global mean_Fmax
        global mean_apce

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()
        #apce用这个17×17的就行了
        response_17 = responses


        # upsample responses and penalize scale changes 上采样至272
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response_17 = response_17[scale_id]

        # 绘制响应图
        # plt.imshow(response_17, cmap='viridis')  # 使用 'viridis' 颜色映射
        # plt.colorbar()  # 添加颜色条
        #
        # plt.savefig('D:/aaa/plt/plot_{0}.png'.format(flag))
        # plt.show()
        # plt.close()
        #####

        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)


        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        # -------------------
        curr_replace, curr_apce, mean_apce = update_model(response_17)
        if curr_replace == 1 and flag % 10 == 1:
            # exemplar image(模板)
            self.avg_color = np.mean(img, axis=(0, 1))  # 计算颜色通道的均值，用于填充
            z = ops.crop_and_resize(
                img, self.center, self.z_sz,
                out_size=self.cfg.exemplar_sz,
                border_value=self.avg_color)  # 获取样本图像的patch

            # exemplar features
            z = torch.from_numpy(z).to(
                self.device).permute(2, 0, 1).unsqueeze(0).float()  # 将patch的channel放到最前，并在channel前加1维
            kernel_new = self.net.backbone(z)  # 将patch送入backbone， 输出的feature作为kernel(核)

            proportion = 1/((self.target_sz[0]*self.target_sz[1])/(img.shape[0]*img.shape[1]))
            if proportion <= 100:
                merge_factor = (0.00049*(proportion**2) + 0.1 )*(curr_apce/(10*mean_apce))
            else:
                merge_factor = 0.2

            self.kernel = (1 - merge_factor) * self.kernel + merge_factor * kernel_new  # 这个0.1的系数是可以改的

            print("更换模板%d" % flag)
            print(merge_factor)
            # -------------------------------------

        flag += 1
        return box

    def track(self, img_files, box, visualize=False):
        global sum_apce
        global sum_Fmax
        global mean_apce
        global mean_Fmax
        global counts
        global flag
        global apce_list
        global Fmax_list
        global f_count

        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        #--
        f_count = frame_num

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
                #参数清零
                sum_apce = 0
                sum_Fmax = 0
                mean_Fmax = 0
                mean_apce = 0
                counts = 1
                flag = 2
                apce_list = []
                Fmax_list = []
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        # ---------------------

        # 将输出列表转换为 DataFrame

        # df = pd.DataFrame(apce_list, columns=['Column_Name'])
        # df1 = pd.DataFrame(Fmax_list, columns=['Column_Name'])
        # # 定义要保存的文件路径和文件名
        # file_path = 'D:/AWork/'
        # file_name = 'apce_value.xlsx'
        # file_name1 = 'Fmax_value.xlsx'
        #
        # # 将 DataFrame 保存为 Excel 文件
        # df.to_excel(file_path + file_name, index=False)
        # df1.to_excel(file_path + file_name1, index=False)
        #
        # print(f"文件已保存至 {file_path + file_name}")

        # ---------------------

        return boxes, times

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels
