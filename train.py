from __future__ import absolute_import

import os
from got10k.datasets import *

import sys
sys.path.append('D:/2/program/SiamFC/siamfc')  # 根据实际路径调整

# from siamfc import TrackerSiamFC   # 改变算法模型
from siamfc_muban_kalman import TrackerSiamFC


if __name__ == '__main__':
    root_dir = os.path.expanduser('D:/aaa/full_data/train_data')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
