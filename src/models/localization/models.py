import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import numpy as np

from src.models.localization.unet import Unet
from src.models.resnet.resnet import ResNetEncoder

class UConv(nn.Module):
    def __init__(self, w_size, num_rot, w_ch, angle_ch):
        super().__init__()
        self.q = Unet(w_size, w_ch, w_ch)
        self.k = copy.deepcopy(self.q)
        self.output = nn.Sequential(nn.Conv2d(num_rot, num_rot, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(num_rot),
                                     nn.ReLU(),
                                     nn.Conv2d(num_rot, 1, kernel_size=3, stride=1, padding=1))
        self.angle_pred = ResNetEncoder(num_rot,spatial_shape=(w_size,w_size), feature_dim=w_size, normalize_visual_inputs=False)
        output_size = np.prod(self.angle_pred.output_shape[0] * (self.angle_pred.output_shape[1]+1)**2)
        self.output2 = nn.Sequential(nn.Linear(output_size, w_size),
                                     nn.ReLU(),
                                     nn.Linear(w_size, angle_ch))

    def forward(self, w, target_embed, rotator):
        #w = F.normalize(w,dim=1)#
        #target_embed = F.normalize(target_embed,dim=1)#)
        key = self.k(w)
        query = self.q(target_embed)
        pivot = [query.shape[-1]//2,query.shape[-1]//2]
        query = torch.stack(rotator(query, pivot),1)

        B, C, H, W = key.shape
        o = F.conv2d(key.reshape(1, B * C, H, W),
                     query.view(B*len(rotator.angles), C, query.size(3), query.size(4)),
                     padding=query.size(3)//2,
                     groups=B)
        o = o.view(B, len(rotator.angles), o.size(2), o.size(3))
        position = self.output(o)
        pred_angle = self.angle_pred(o).view(B, -1)
        angle = self.output2(pred_angle)
        return position, angle
