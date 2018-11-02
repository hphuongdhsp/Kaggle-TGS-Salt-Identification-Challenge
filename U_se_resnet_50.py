#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:16:46 2018

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
import se_resnet_50
from torch.nn import functional as F
from torchsummary import summary

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


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)



class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = SynchronizedBatchNorm2d(out_channels)


    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels ):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x ):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        return x

class SEResNet50UNet(nn.Module):

    def __init__(self ):
        super().__init__()
        se_resnet = se_resnet_50.se_resnet50(num_classes=1000, pretrained='imagenet')
        self.conv1 = nn.Sequential(
              ConvBn2d(3, 64, kernel_size=3, stride=2, padding=1),
              nn.ReLU(inplace=True),
        ) # out 128

        self.encoder1 = se_resnet.layer1  #out 256
        self.encoder2 = se_resnet.layer2  #out 256
        self.encoder3 = se_resnet.layer3  #out 1024
        self.encoder4 = se_resnet.layer4  #out = 512*4 = 2048

        self.center = nn.Sequential(
            ConvBn2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = Decoder(2048+256, 512, 256)
        self.decoder4 = Decoder(1024+256, 512, 256)
        self.decoder3 = Decoder( 512+256, 256,  64)
        self.decoder2 = Decoder( 256+ 64, 128, 128)
        self.decoder1 = Decoder( 128    , 128,  32)


        self.logit    = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  1, kernel_size=1, padding=0)
        )



    def forward(self, x):
        #batch_size,C,H,W = x.shape
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) #64xH/2xW/2 maybe not max pool

        e2 = self.encoder1(x)   #256X H/2XW/2
        e3 = self.encoder2(e2)  #512XH/4XW/4
        e4 = self.encoder3(e3)  #1024X H/8XW/8
        e5 = self.encoder4(e4)  #2048X H/16XW/16


        #f = F.max_pool2d(e5, kernel_size=2, stride=2 )  #; print(f.size())
        #f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)#False
        #f = self.center(f)                       #; print('center',f.size())
        f = self.center(e5) #256XH/16XW/16
         
        f = self.decoder5(torch.cat([f, e5], 1)) # 256xH/8XW/8
        f = self.decoder4(torch.cat([f, e4], 1)) # 256x512XH/4XW/4
        f = self.decoder3(torch.cat([f, e3], 1)) # 64X H/2XW/2
        f = self.decoder2(torch.cat([f, e2], 1)) # 128
        f = self.decoder1(f)                     # 32

        f = F.dropout2d(f, p=0.20)
        logit = self.logit(f)                     #; print('logit',logit.size())
        return logit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SEResNet50UNet().to(device)
summary(model, (3, 128, 128))
