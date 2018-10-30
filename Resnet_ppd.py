#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:30:30 2018

@author: ai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 01:00:43 2018

@author: ai
"""


from collections import OrderedDict
from torchsummary import summary
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
        #self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = self.selu(x)
        return x
class Conv2BN1(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out)
        #self.selu = nn.SELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x     

class Decoder(nn.Module):
    def __init__(self, channels_e, channels_d, out_channels,is_deconv,reduction,block):
        super(Decoder,self).__init__()


        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        #self.conv1 = Conv2BN(out_channels,  int(out_channels/2) )
        #self.conv2 = Conv2BN(out_channels,  int(out_channels/2) )
        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(channels_e + channels_d, out_channels),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(channels_e + channels_d, out_channels),
                ConvRelu(out_channels, out_channels)
            )
        self.conv1 = Conv2BN1(channels_e,channels_e) 
        self.up=block
        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.scse= ModifiedSCSEBlock(out_channels, reduction)
    def forward(self, x, e=None):
        #x = F.upsample(x,scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            #e = self.conv1(e)
            x=torch.cat([x,e],1)
        if self.up:
            x = self.block(x)
        x = self.scse(x)
        return x

class Hypercolumn(nn.Module):
    def __init__(self):
        super(Hypercolumn,self).__init__()
    def forward(self,x2,X,x3,x4,x5):
        x = torch.cat((x2,X,
                      F.upsample(x3,scale_factor=2,  mode= 'bilinear', align_corners=False),
                      F.upsample(x4,scale_factor=4,  mode= 'bilinear', align_corners=False),
                      F.upsample(x5,scale_factor=8,  mode= 'bilinear', align_corners=False)
                      #F.upsample(x5,scale_factor=16,  mode= 'bilinear', align_corners=False)
                      ),1)
        return x

class CenterBlockLinkNet(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(8, 8), rate=(3, 5)):
        super(CenterBlockLinkNet, self).__init__()
        self.gave_pool = nn.Sequential(OrderedDict([("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                                                    ("conv1x1", nn.Conv2d(in_chs, out_chs,
                                                                          kernel_size=1, stride=1, padding=0,
                                                                          groups=1, bias=False, dilation=1)),
                                                    ("up0", nn.Upsample(size=feat_res, mode='bilinear')),
                                                    ("bn0", nn.BatchNorm2d(num_features=out_chs))]))

        self.conv3x3 = nn.Sequential(OrderedDict([("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                        stride=1, padding=1, bias=False,
                                                                        groups=1, dilation=1)),
                                                   ("bn3x3", nn.BatchNorm2d(num_features=out_chs))]))

        self.vortex_bra1 = nn.Sequential(OrderedDict([("avg_pool", nn.AvgPool2d(kernel_size=rate[0], stride=1,
                                                                                padding=int((rate[0]-1)/2), ceil_mode=True)),
                                                      ("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                            stride=1, padding=rate[0], bias=False,
                                                                            groups=1, dilation=rate[0])),
                                                      ("bn3x3", nn.BatchNorm2d(num_features=out_chs))]))

        self.vortex_bra2 = nn.Sequential(OrderedDict([("avg_pool", nn.AvgPool2d(kernel_size=rate[1], stride=1,
                                                                                padding=int((rate[1]-1)/2), ceil_mode=True)),
                                                      ("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3,
                                                                            stride=1, padding=rate[1], bias=False,
                                                                            groups=1, dilation=rate[1])),
                                                      ("bn3x3", nn.BatchNorm2d(num_features=out_chs))]))


        self.vortex_catdown = nn.Sequential(OrderedDict([("conv_down", nn.Conv2d(4 * out_chs, out_chs, kernel_size=1,
                                                                                 stride=1, padding=0, bias=False,
                                                                                 groups=1, dilation=1)),
                                                         ("bn_down", nn.BatchNorm2d(num_features=out_chs)),
                                                         ("dropout", nn.Dropout2d(p=0.2, inplace=True))]))
        #self.sesc = ModifiedSCSEBlock(out_chs, reduction=4)

        #self.upsampling = nn.Upsample(size=(int(feat_res[0] * up_ratio), int(feat_res[1] * up_ratio)), mode='bilinear')
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = torch.cat((self.gave_pool(x),
                         self.conv3x3(x),
                         self.vortex_bra1(x),
                         self.vortex_bra2(x)
                         ), 1)

        out = self.vortex_catdown(out)
        return self.relu(out)

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3,is_deconv=True, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)
        #self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
        #                       bias=False),
        #            nn.BatchNorm2d(64),
        #            nn.ReLU(inplace=True)
        #             )
        self.encoder1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        )  #128x128x65
        self.encoder2 = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        resnet.layer1,
                        ) # 64x64x64
        self.encoder3 = resnet.layer2 #128x32x32
        self.encoder4 = resnet.layer3 #256x16x16
        self.encoder5 = resnet.layer4 #512X8x8
        self.hypercolumn=Hypercolumn()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
         #512x
        ## center
        self.center =CenterBlockLinkNet(512, 512, feat_res=(8, 8), rate=(3, 5))  #256x8x8
        self.transform=nn.Sequential(Conv2BN1(64,  64 ),
                                     ModifiedSCSEBlock(64, 2))
        # Decoder
        self.decoder5 = Decoder(filters[3], filters[3],filters[0],is_deconv,4,block=True)#32x 8x8
        self.decoder4 = Decoder(filters[2], filters[0],filters[0],is_deconv,4,block=True)#32x16x16
        self.decoder3 = Decoder(filters[1], filters[0],filters[0],is_deconv,4,block=True)#32x32x32
        self.decoder2 = Decoder(filters[0], filters[0],filters[0],is_deconv,4,block=True)#32x64x64
        self.decoder1 = Decoder(filters[0], filters[0],filters[0],is_deconv,4,block=False)#32128x128

        # logit
        self.logit =nn.Sequential(
            nn.Conv2d(320 ,32,kernel_size=3,padding=1),#nn.Conv2d(320,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,1,kernel_size=1,padding=0)
        ) 

        # Final Classifier
        #self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):

        # Encoder
 #64x64x64   
        e1 = self.encoder1(x)#64x64x64
        e2 = self.encoder2(e1)#128x32x32
        e3 = self.encoder3(e2)#256x16x16
        e4 = self.encoder4(e3)#512x8x8
        e5 = self.encoder5(e4)

        # center 
        f = self.center(e5)   #512x8x8

        # Decoder with Skip Connections
        d5 = self.decoder5(f,e5) #64x16x16
        d4 = self.decoder4(d5,e4)#64x32x32 
        d3 = self.decoder3(d4,e3)#64x64x64
        d2 = self.decoder2(d3,e2)#64x128,128
#        d1 = self.decoder1(d2,e1)
        
        X  = self.transform(e1)

        # Final Classification
        f = self.hypercolumn(X,d2,d3,d4,d5)
        
        
        f=F.dropout2d(f,p=0.2)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = CenterBlockLinkNet( 3, 3, feat_res=(8, 8), rate=(3, 5)).to(device)
summary(model1, (3, 8, 8))
model = LinkNet34(num_classes=1, num_channels=3,is_deconv=True, pretrained=True).to(device)
summary(model, (3, 128, 128))


