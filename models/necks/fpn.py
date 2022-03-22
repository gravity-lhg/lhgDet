# Author: Haoguang Liu
# Data: 2022.3.21 18:50 PM
# Email: 1052979481@qq.com
# Github: https://github.com/gravity-lhg

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    '''only for resnet50,101,152 and resnet_cbam50,101,152'''
    def __init__(self, features=256, use_p5=False):
        '''
        args: 
            features: output channels
            use_p5: use P5 to generate P6(True), use C5 to generate P6(False)
        '''
        super(FPN, self).__init__()
        self.proj_5 = nn.Conv2d(2048, features, kernel_size=1)
        self.proj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.proj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.smooth_conv_4 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.smooth_conv_3 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        if use_p5:
            self.proj_6 = nn.Conv2d(features, features, kernel_size=3, stride=2, padding=1)
        else:
            self.proj_6 = nn.Conv2d(2048, features, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)
        self.proj_7 = nn.Conv2d(features, features,kernel_size=3, stride=2, padding=1)
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, input):
        src, target = input
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        P5 = self.proj_5(C5)
        P4 = self.smooth_conv_4(self.proj_4(C4) + self.upsamplelike([P5, C4]))
        P3 = self.smooth_conv_3(self.proj_3(C3) + self.upsamplelike([P4, C3]))
        if self.use_p5:
            P6 = self.proj_6(P5)
        else:
            P6 = self.proj_6(C5)
        P7 = self.proj_7(self.relu(self.bn(P6)))
        return [P3, P4, P5, P6, P7]

if __name__=='__main__':
    # for print network
    net = FPN()
    # print(net)

    # for test fpn network
    inputs = (torch.rand(3, 512, 224, 224).clone().detach(), 
              torch.rand(3, 1024, 112, 112).clone().detach(), 
              torch.rand(3, 2048, 56, 56).clone().detach())
    outputs = net(inputs)
    print(outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape, outputs[4].shape)