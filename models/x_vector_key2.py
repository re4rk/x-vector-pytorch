#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""


import torch.nn as nn
import torch.nn.functional as F
from models.tdnn import TDNN
import torch


class X_vector(nn.Module):
    def __init__(self, input_dim = 40, num_classes=8):
        super(X_vector, self).__init__()
        self.tdnnLayer = nn.Sequential(
            TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5),
            TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5),
            TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5),
            TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5),
            TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        )

        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)

        self.layer1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        #### Frame level Pooling
        tdnnLayer_out = self.tdnnLayer(inputs)

        mean = torch.mean(tdnnLayer_out,1)
        std = torch.var(tdnnLayer_out,1)
        stat_pooling = torch.cat((mean,std),1)

        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)

        x_vec = x_vec.clone()
        out = self.layer1(x_vec)
        out = out + x_vec
        out = self.layer2(out)
        return out, x_vec