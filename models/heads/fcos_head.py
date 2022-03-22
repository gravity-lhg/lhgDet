# Author: Haoguang Liu
# Data: 2022.3.22 22:45 PM
# Email: 1052979481@qq.com
# Github: https://github.com/gravity-lhg

import torch
import torch.nn as nn
import math

class ScaleExp(nn.Module):
    ''''''
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
    def forward(self, x):
        return torch.exp(x * self.scale)

class ClsCntRegHead(nn.Module):
    '''Heads of Classification, Center-ness and Regression for FCOS network'''
    def __init__(self, channels, class_num, GroupNorm=True, cnt_on_cls=True, prior=0.01):
        '''
        args:
            channels: number of channels held in the heads
            class_num: number of class in dataset
            GroupNorm: whether to use group normalization
            cnt_on_cls: whether Center-ness extends from the Classification network
            prior: for initialize the final conv layer of the classification subnet
        '''
        super(ClsCntRegHead, self).__init__()
        self.cnt_on_cls = cnt_on_cls
        cls_branch = []
        reg_branch = []
        for _ in range(4):
            # create part of classification subnet
            cls_branch.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
            if GroupNorm:
                cls_branch.append(nn.GroupNorm(32, channels))
            cls_branch.append(nn.ReLU(inplace=True))

            # create part of regression subnet
            reg_branch.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True))
            if GroupNorm:
                reg_branch.append(nn.GroupNorm(32, channels))
            reg_branch.append(nn.ReLU(inplace=True))

        self.cls_subnet = nn.Sequential(*cls_branch)
        self.reg_subnet = nn.Sequential(*reg_branch)

        self.cls_logits_conv = nn.Conv2d(channels, class_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.cnt_logits_conv = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.reg_predict_conv = nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1, bias=True)

        self.apply(self.init_conv_RandomNormal)

        nn.init.constant_(self.cls_logits_conv.bias, -math.log((1 - prior) / prior))
        # ModuleList like a list, used to add similar or repeated layers to the network
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        '''inputs = [P3 ~ P7]'''
        cls_logits = []
        cnt_logits = []
        reg_predict = []
        for index, layer in enumerate(inputs):
            cls_subnet_out = self.cls_subnet(layer)
            reg_subnet_out = self.reg_subnet(layer)

            cls_logits.append(self.cls_logits_conv(cls_subnet_out))
            if self.cnt_on_cls:
                cnt_logits.append(self.cnt_logits_conv(cls_subnet_out))
            else:
                cnt_logits.append(self.cnt_logits_conv(reg_subnet_out))
            reg_predict.append(self.scale_exp[index](self.reg_predict_conv(reg_subnet_out)))
        return cls_logits, cnt_logits, reg_predict

if __name__=='__main__':
    # for print network
    net = ClsCntRegHead(256, 8)
    # print(net)

    # for test heads subnet
    inputs = [torch.rand(3, 256, 224, 224).clone().detach(),
              torch.rand(3, 256, 112, 112).clone().detach(),
              torch.rand(3, 256, 56, 56).clone().detach()]
    cls_logits, cnt_logits, reg_predict = net(inputs)
    print(cls_logits[0].shape, cnt_logits[1].shape, reg_predict[2].shape)