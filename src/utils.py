# Utility functions for metrics and visualizations
import os
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import imageio
import SimpleITK as sitk
from medpy import metric
from einops import repeat
from scipy.ndimage import zoom
from datetime import datetime
from torch.optim.lr_scheduler import LambdaLR
from icecream import ic


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = 1.0 * (input_tensor == i)
            temp_prob[input_tensor == -100] = -100
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        mask = (target != -100)
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'Predict {inputs.size()} & Target {target.size()} mismatch'
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        return metric.binary.dc(pred, gt)
    elif pred.sum() == 0 and gt.sum() == 0:
        return 1
    else:
        return 0


class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup followed by cosine decay learning rate scheduler.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * self.cycles * 2.0 * progress)))
