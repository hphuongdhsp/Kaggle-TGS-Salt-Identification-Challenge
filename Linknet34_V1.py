#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:15:06 2018

@author: ai
"""

from torch import nn
import torch
from torchvision import models
from torch.nn import functional as F
from torchsummary import summary
from self_attention import BaseOC_Context_Module
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out)

    def forward(self, x):
        x = self.conv(x)

        x = self.activation(x)
        x = self.bn(x)
        return x

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)





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

class Hypercolumn(nn.Module):
    def __init__(self):
        super(Hypercolumn,self).__init__()
    def forward(self,e1,x1,x2,x3,x4):
        x = torch.cat([e1,x1,
                      F.upsample(x2,scale_factor=2,  mode= 'bilinear', align_corners=False),
                      F.upsample(x3,scale_factor=4,  mode= 'bilinear', align_corners=False),
                      F.upsample(x4,scale_factor=8,  mode= 'bilinear', align_corners=False)
                      ],1)
        return x

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
        self.selu = nn.SELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
class DecoderBlock(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
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
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2BN(in_channels, middle_channels),
                Conv2BN(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3,  pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)


        self.encoder1= nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,
                               bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True) #128x64
                     )
        self.encoder2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                      resnet.layer1) # 64x64
        self.encoder3 = nn.Sequential(resnet.layer2) # 32x128
        self.encoder4 = nn.Sequential(resnet.layer3) # 16x256
        self.encoder5 = nn.Sequential(resnet.layer4) # 8x512
        #self.gate1    = NONLocalBlock2D(64, mode='gaussian', sub_sample=False, bn_layer=False)
        self.gate2    = NONLocalBlock2D(64, mode='gaussian', sub_sample=False, bn_layer=False)
        self.gate3    = NONLocalBlock2D(128, mode='gaussian', sub_sample=False, bn_layer=False)
        self.gate4    = NONLocalBlock2D(256, mode='gaussian', sub_sample=False, bn_layer=False)
        #self.gate5    = NONLocalBlock2D(512, mode='gaussian', sub_sample=False, bn_layer=False)
        self.center   = BaseOC_Context_Module(512, 512, 128, 128, 0.05, sizes=([1])) #4x512
        # Decoder
        """
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        """
        #self.decoder5 = DecoderBlock(512, 256, 256, is_deconv=False) #8x256
        self.decoder4 = nn.Sequential(DecoderBlock(filters[3]+ filters[3],filters[3], filters[0],is_deconv=False),
                                       ModifiedSCSEBlock(filters[0],2))
        self.decoder3 = nn.Sequential(DecoderBlock(filters[2]+ filters[0],filters[2], filters[0],is_deconv=False),
                                      ModifiedSCSEBlock(filters[0],2))
        self.decoder2 = nn.Sequential(DecoderBlock(filters[1]+ filters[0],filters[1], filters[0],is_deconv=False),
                                      ModifiedSCSEBlock(filters[0],2))
        self.decoder1 = nn.Sequential(DecoderBlock(filters[0]+ filters[0],filters[0], filters[0],is_deconv=False),
                                      ModifiedSCSEBlock(filters[0],2))
        self.hypercolumn=Hypercolumn()
        # Final Classifier
        #self.finaldeconv1 = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, 4, stride=2,padding=1),nn.ReLU(inplace=True))
        self.finalconv2 = nn.Conv2d(320, 32, 3,padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 1, padding=0)
        #self.pool = nn.Sequential(nn.MaxPool2d(2, 2), Conv2BN(512,256),Conv2BN1(256,512))
        self.pool = nn.MaxPool2d(2, 2)
    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x) #64
        e2 = self.encoder2(e1)#64
        e3 = self.encoder3(e2)#128
        e4 = self.encoder4(e3)#256
        e5 = self.encoder5(e4)#512
        
        #poole5=self.pool(e5)
        center = self.center(e5)
        
        #d5 = self.decoder5(torch.cat([center, e5], 1))

        # Decoder with Skip Connections
        d4 = self.decoder4(torch.cat([center, e5], 1))
        d3 = self.decoder3(torch.cat([d4, self.gate4(e4)], 1))
        d2 = self.decoder2(torch.cat([d3, self.gate3(e3)], 1))
        d1 = self.decoder1(torch.cat([d2, self.gate2(e2)], 1))

        # Final Classification

        f2 = self.hypercolumn(e1,d1,d2,d3,d4)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinkNet34(num_classes=1, num_channels=3, pretrained=True).to(device)
summary(model, (3, 128, 128))