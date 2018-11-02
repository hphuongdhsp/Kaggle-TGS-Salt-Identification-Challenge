from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F
from torchsummary import summary

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
        x = self.bn(x)
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

class Hypercolumn(nn.Module):
    def __init__(self):
        super(Hypercolumn,self).__init__()
    def forward(self,e1,x1,x2,x3,x4,x5):
        x = torch.cat([e1,x1,
                      F.upsample(x2,scale_factor=2,  mode= 'bilinear', align_corners=False),
                      F.upsample(x3,scale_factor=4,  mode= 'bilinear', align_corners=False),
                      F.upsample(x4,scale_factor=8,  mode= 'bilinear', align_corners=False),
                      F.upsample(x5,scale_factor=16, mode= 'bilinear', align_corners=False)
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
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)
class Conv2BN(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = self.selu(x)
        return x    
    
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
                                       resnet.layer1,ModifiedSCSEBlock(64,2)) # 64x64
        self.encoder3 = nn.Sequential(resnet.layer2,ModifiedSCSEBlock(128,2)) # 32x128
        self.encoder4 = nn.Sequential(resnet.layer3,ModifiedSCSEBlock(256,2)) # 16x256
        self.encoder5 = nn.Sequential(resnet.layer4,ModifiedSCSEBlock(512,2)) # 8x512
        
        self.center = DecoderBlock(512, 256, 128, is_deconv=False) #4x512
        # Decoder
        """
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        """
        #self.decoder5 = DecoderBlock(512, 256, 256, is_deconv=False) #8x256
        self.decoder4 = nn.Sequential(DecoderBlock(filters[3]+ filters[1],filters[3], filters[0],is_deconv=False),
                                       ModifiedSCSEBlock(filters[0],4))
        self.decoder3 = nn.Sequential(DecoderBlock(filters[2]+ filters[0],filters[2], filters[0],is_deconv=False),
                                      ModifiedSCSEBlock(filters[0],4))
        self.decoder2 = nn.Sequential(DecoderBlock(filters[1]+ filters[0],filters[1], filters[0],is_deconv=False),
                                      ModifiedSCSEBlock(filters[0],4))
        self.decoder1 = nn.Sequential(DecoderBlock(filters[0]+ filters[0],filters[0], filters[0],is_deconv=False),
                                      ModifiedSCSEBlock(filters[0],4))
        self.hypercolumn=Hypercolumn()
        # Final Classifier
        #self.finaldeconv1 = nn.Sequential(nn.ConvTranspose2d(filters[0], 32, 4, stride=2,padding=1),nn.ReLU(inplace=True))
        self.finalconv2 = nn.Conv2d(448, 32, 3,padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 1, padding=0)
        self.pool = nn.Sequential(nn.MaxPool2d(2, 2))

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x) #64
        e2 = self.encoder2(e1)#64
        #e22 = F.dropout2d(e2, 0.2)
        e3 = self.encoder3(e2)#128
        #e33 = F.dropout2d(e3, 0.2)
        e4 = self.encoder4(e3)#256
        #e44 = F.dropout2d(e4, 0.2)
        e5 = self.encoder5(e4)#512
        #e55 = F.dropout2d(e5, 0.2)
        poole5=self.pool(e5)
        center = self.center(poole5)        
        #d5 = self.decoder5(torch.cat([center, e5], 1))

        # Decoder with Skip Connections
        d4 = self.decoder4(torch.cat([center, e5], 1))
        d3 = self.decoder3(torch.cat([d4, e4], 1))
        d2 = self.decoder2(torch.cat([d3, e3], 1))
        d1 = self.decoder1(torch.cat([d2, e2], 1))

        # Final Classification

        f2 = self.hypercolumn(e1,d1,d2,d3,d4,center)
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