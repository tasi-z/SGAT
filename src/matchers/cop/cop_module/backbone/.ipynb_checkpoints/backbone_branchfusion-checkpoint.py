import torch.nn as nn
import torch.nn.functional as F
import torch
from .repvgg import create_RepVGG,create_RepVGG_6
from einops.einops import rearrange

def get_stage_output_channels(stage):
    # 获取 stage 中每个 RepVGGBlock 的输出通道数
    for block in reversed(stage):
        if hasattr(block, 'rbr_dense') and isinstance(block.rbr_dense, nn.Sequential):
            for layer in block.rbr_dense:
                if isinstance(layer, nn.Conv2d):
                    return layer.out_channels
        if hasattr(block, 'rbr_1x1') and isinstance(block.rbr_1x1, nn.Sequential):
            for layer in block.rbr_1x1:
                if isinstance(layer, nn.Conv2d):
                    return layer.out_channels
    raise ValueError("No Conv2d layer found in the RepVGGBlock")

class RepVGG_8_1_align(nn.Module):
    """
    RepVGG backbone, output resolution are 1/8 and 1.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        Iv_backbone = create_RepVGG_6(False)
        V_backbone = create_RepVGG_6(False)
        self.layer0, self.layer1, self.layer2, self.layer3 = Iv_backbone.stage0, Iv_backbone.stage1, Iv_backbone.stage2, Iv_backbone.stage3
        self.vlayer0, self.vlayer1, self.vlayer2, self.vlayer3 = V_backbone.stage0, V_backbone.stage1, V_backbone.stage2, V_backbone.stage3
        c1 = get_stage_output_channels(self.layer1)
        c2 = get_stage_output_channels(self.layer2)
        c3 = get_stage_output_channels(self.layer3)
        self.cat_filter1 = nn.Sequential(
            nn.Linear(2 * c1, 2 * c1),
            nn.LayerNorm(2 * c1, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * c1, c1),
        )
        self.cat_filter2 = nn.Sequential(
            nn.Linear(2 * c2, 2 * c2),
            nn.LayerNorm(2 * c2, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * c2, c2),
        )
        self.cat_filter3 = nn.Sequential(
            nn.Linear(2 * c3, 2 * c3),
            nn.LayerNorm(2 * c3, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * c3, c3),
        )

        for layer in [self.layer0, self.layer1, self.layer2, self.layer3,self.vlayer0, self.vlayer1, self.vlayer2, self.vlayer3]:
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
        x1 = out
        for module in self.layer2:
            out = module(out) # 1/4
        x2 = out
        for module in self.layer3:
            out = module(out) # 1/8
        x3 = out

        vout = self.vlayer0(x) # 1/2
        for module in self.vlayer1:
            vout = module(vout) # 1/2
        vx1 = vout
        for module in self.vlayer2:
            vout = module(vout) # 1/4
        vx2 = vout
        for module in self.vlayer3:
            vout = module(vout) # 1/8
        vx3 = vout

        # Concatenate and rearrange before passing to cat_filter1
        fusion_x1_input = torch.cat([x1, vx1], dim=1)
        fusion_x1_input = rearrange(fusion_x1_input, 'n c h w -> n (h w) c')
        fusion_x1_flat = self.cat_filter1(fusion_x1_input)
        fusion_x1 = rearrange(fusion_x1_flat, 'n (h w) c -> n c h w', h=x1.size(2), w=x1.size(3))

        # Repeat the process for x2, vx2 and x3, vx3
        fusion_x2_input = torch.cat([x2, vx2], dim=1)
        fusion_x2_input = rearrange(fusion_x2_input, 'n c h w -> n (h w) c')
        fusion_x2_flat = self.cat_filter2(fusion_x2_input)
        fusion_x2 = rearrange(fusion_x2_flat, 'n (h w) c -> n c h w', h=x2.size(2), w=x2.size(3))

        fusion_x3_input = torch.cat([x3, vx3], dim=1)
        fusion_x3_input = rearrange(fusion_x3_input, 'n c h w -> n (h w) c')
        fusion_x3_flat = self.cat_filter3(fusion_x3_input)
        fusion_x3 = rearrange(fusion_x3_flat, 'n (h w) c -> n c h w', h=x3.size(2), w=x3.size(3))

        return {'sfeats_c': x3, 'tfeats_c': vx3,'sfeats_f': None, 'feats_c':fusion_x3,'feats_x2': fusion_x2,'feats_x1': fusion_x1}
