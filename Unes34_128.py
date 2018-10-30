#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 18:29:32 2018

@author: ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 13:09:53 2018

@author: ai
"""

from torch import nn
import torch
from torchvision import models
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


class ModifiedSCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ModifiedSCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)

        spa_se = self.spatial_se(x)
        return torch.mul(torch.mul(x, chn_se), spa_se)

class Conv2BN(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x    

class Decoder(nn.Module):
    def __init__(self, in_channels, channel, out_channels,reduction):
        super(Decoder,self).__init__()


        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = Conv2BN(in_channels,  channel )
        self.conv2 = Conv2BN(channel, out_channels )

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.scse= ModifiedSCSEBlock(out_channels, reduction)
    def forward(self, x, e=None):
        x = F.upsample(x,scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x=torch.cat([x,e],1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.scse(x)
        return x
    


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        #self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        ## center
        self.center =nn.Sequential(
            Conv3BN(512,512, bn=True),
            Conv3BN(512,256, bn=True),
            nn.MaxPool2d(kernel_size=2,stride=2
            )
        )
        # Decoder
        self.decoder5 = Decoder(filters[3]+filters[2], filters[3],filters[0],2)
        self.decoder4 = Decoder(filters[2]+filters[0], filters[2],filters[0],2)
        self.decoder3 = Decoder(filters[1]+filters[0], filters[1],filters[0],2)
        self.decoder2 = Decoder(filters[0]+filters[0], filters[0],filters[0],2)
        self.decoder1 = Decoder(filters[0], int(filters[0]/2) ,filters[0],2)

        # logit
        self.logit =nn.Sequential(
            nn.Conv2d(320,64,kernel_size=3,padding=1),#nn.Conv2d(320,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,kernel_size=1,padding=0)
        ) 

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):

        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        #x = self.firstmaxpool(x)   
        e2 = self.encoder1(x)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)

        # center 
        f = self.center(e5)

        # Decoder with Skip Connections
        d5 = self.decoder5(f,e5)
        d4 = self.decoder4(d5,e4) 
        d3 = self.decoder3(d4,e3)
        d2 = self.decoder2(d3,e2)
        d1 = self.decoder1(d2)
        
        

        # Final Classification
        f = torch.cat((
            d1,
            F.upsample(d2,scale_factor=2,  mode= 'bilinear', align_corners=False),
            F.upsample(d3,scale_factor=4,  mode= 'bilinear', align_corners=False),
            F.upsample(d4,scale_factor=8,  mode= 'bilinear', align_corners=False),
            F.upsample(d5,scale_factor=16, mode= 'bilinear', align_corners=False)
        ),1)
        
        #f=F.dropout2d(f, p=0.5)
        logit = self.logit(f)
        return logit


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x




