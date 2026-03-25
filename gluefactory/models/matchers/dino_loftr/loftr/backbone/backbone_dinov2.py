import torch.nn as nn
import torch.nn.functional as F
from .repvgg import create_RepVGG_rgb
from .dino import DINO
from .lift import LiFT
import torch
def build_backbone(config):
    if config['backbone_type'] == 'RepVGG':
        if config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return DINO_align(config['backbone'])
        else:
            raise ValueError(f"LOFTR.ALIGN_CORNER {config['align_corner']} not supported.")
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, bn=True):
        super().__init__()
        self.conv = conv3x3(in_planes, planes, stride)
        self.bn = nn.BatchNorm2d(planes) if bn is True else None
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        if self.bn:
            y = self.bn(y)
        y = self.act(y)
        return y

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

class RepVGG_8_1_align(nn.Module):
    """
    RepVGG backbone, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        backbone = create_RepVGG_rgb(False)

        self.layer0, self.layer1, self.layer2, self.layer3 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3

        for layer in [self.layer0, self.layer1, self.layer2, self.layer3]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layer0(x) # 1/2
        for module in self.layer1:
            out = module(out) # 1/2
        x1 = out #64
        for module in self.layer2:
            out = module(out) # 1/4
        x2 = out #128
        for module in self.layer3:
            out = module(out) # 1/8
        x3 = out #256
        
        return {'feats_1_4': x2, 'feats_1_2': x1,'feats_1_8':x3}
    
class DINO_align(nn.Module):
    """
    RepVGG backbone, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        self.dino_backbone=DINO(dino_name="dinov2",model_name="vitb14",output="dense")
        self.dino_backbone.requires_grad_(False)
        self.dino_backbone.eval()
        self.dino_backbone.to(torch.float16)
        self.dino_upscale=LiFT(in_channels=768,patch_size=14,pre_shape=False,post_shape=False)
        self.fine_backbone=RepVGG_8_1_align(config)
        self.conv_layer256= nn.Conv2d(in_channels=768+256, out_channels=256, kernel_size=1)

        # self.recon_convt3 = convt_bn_relu(in_channels=768, out_channels=256, kernel_size=3, stride=2, padding=1,
        #                             output_padding=1)
        # self.recon_convt2 = convt_bn_relu(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
        #                             output_padding=1)
        #降维成256
        # self.conv_layer256= nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1)
        # self.conv_layer128= nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        # self.conv_layer64= nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)

    def forward(self, x,coarse_size):
        b,_,h,w=x.shape
        # 1/14
        dino_feat= self.dino_backbone(x.to(torch.float16)).to(torch.float32)
        coarse_h0,coarse_w0=coarse_size
        new_coarse_h0,new_coarse_w0=coarse_h0//4*4,coarse_w0//4*4
        fine_feat=self.fine_backbone(x)
        fine_feat_1_4=fine_feat['feats_1_4']
        fine_feat_1_2=fine_feat['feats_1_2']
        fine_feat_1_8=fine_feat['feats_1_8']
        
        dino_feat_up1=self.dino_upscale(x,dino_feat)
        # dino_feat_up1_8=F.interpolate(dino_feat_up1,size=(coarse_h0,coarse_w0),mode='bilinear')
        dino_feat_up1_8=F.interpolate(dino_feat_up1,size=(fine_feat_1_8.shape[2],fine_feat_1_8.shape[3]),mode='bilinear')

        dino_feat_up1_8=self.conv_layer256(torch.cat([dino_feat_up1_8,fine_feat_1_8],dim=1))
        if dino_feat_up1_8.shape[2]!=new_coarse_h0 or dino_feat_up1_8.shape[3]!=new_coarse_w0:
            dino_feat_up1_8=F.interpolate(dino_feat_up1_8,size=(new_coarse_h0,new_coarse_w0),mode='bilinear')
        # out = self.layer0(x) # 1/2
        # for module in self.layer1:
        #     out = module(out) # 1/2
        # x1 = out
        # for module in self.layer2:
        #     out = module(out) # 1/4
        # x2 = out
        # for module in self.layer3:
        #     out = module(out) # 1/8
        # x3 = out
                
        return {'dino_feat':dino_feat,'feats_c': dino_feat_up1_8, 'feats_f': None, 'feats_x2': fine_feat_1_4, 'feats_x1': fine_feat_1_2}
