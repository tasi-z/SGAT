from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from loguru import logger

from .dino import DINO
from .dino_loftr.loftr.backbone.backbone_dinov2 import build_backbone
from .dino_loftr.loftr.loftr_module import LocalFeatureTransformer
from .dino_loftr.utils.misc import detect_NaN
from ..utils.util import (
    center_padding,
    draw_patch_match,
    draw_torch_image,
    get_matchable_map,
    get_max_cos_similarity,
    get_self_similarity,
    tokens_to_output,
)


class COP(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()
        self.config = config
        self.profiler = profiler

        self.backbone = build_backbone(config)
        self.loftr_coarse = LocalFeatureTransformer(config)
        self.classifier = nn.Linear(config["coarse"]["d_model"], 1)
        self.get_SPA = True
        if config["weights"] is not None:
            state_dict = torch.load(config["weights"], map_location="cuda")
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def get_SPAmask(self, data, feat_0, feat_1):
        img0_target_h, img0_target_w = data["hw0_i"]
        img1_target_h, img1_target_w = data["hw1_i"]
        max_similarity_map_list0 = []
        max_similarity_map_list1 = []
        matchable_map_list0 = []
        matchable_map_list1 = []
        range_list = [16, 32, 64]
        for p in range_list:
            feats0 = torch.nn.functional.interpolate(
                feat_0,
                (int(img0_target_h / p), int(img0_target_w / p)),
                mode="bilinear",
                align_corners=True,
            )
            feats1 = torch.nn.functional.interpolate(
                feat_1,
                (int(img1_target_h / p), int(img1_target_w / p)),
                mode="bilinear",
                align_corners=True,
            )
            max_similarity_map0 = get_self_similarity(feats0)
            max_similarity_map0_up = torch.nn.functional.interpolate(
                max_similarity_map0,
                (img0_target_h, img0_target_w),
                mode="bilinear",
                align_corners=True,
            )
            max_similarity_map_list0.append(max_similarity_map0_up)
            max_similarity_map1 = get_self_similarity(feats1)
            max_similarity_map1_up = torch.nn.functional.interpolate(
                max_similarity_map1,
                (img1_target_h, img1_target_w),
                mode="bilinear",
                align_corners=True,
            )
            max_similarity_map_list1.append(max_similarity_map1_up)

        ave_max_similarity_map0 = torch.zeros(max_similarity_map_list0[0].shape).cuda()
        for max_similarity_map in max_similarity_map_list0:
            ave_max_similarity_map0 += max_similarity_map
        ave_max_similarity_map0 = ave_max_similarity_map0 / len(max_similarity_map_list0)

        ave_max_similarity_map1 = torch.zeros(max_similarity_map_list1[0].shape).cuda()
        for max_similarity_map in max_similarity_map_list1:
            ave_max_similarity_map1 += max_similarity_map
        ave_max_similarity_map1 = ave_max_similarity_map1 / len(max_similarity_map_list1)

        ave_max_similarity_map0 = 1 - ave_max_similarity_map0
        ave_max_similarity_map1 = 1 - ave_max_similarity_map1

        return ave_max_similarity_map0, ave_max_similarity_map1

    def forward(self, view0, view1):
        data = {
            "image0": view0["image"],
            "image1": view1["image"],
            "image0_rgb": view0["image"],
            "image1_rgb": view1["image"],
        }
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )
        if "coarse_h0" not in data:
            data["coarse_h0"] = data["hw0_i"][0] // self.config["resolution"][0]
            data["coarse_w0"] = data["hw0_i"][1] // self.config["resolution"][0]
            data["coarse_h1"] = data["hw1_i"][0] // self.config["resolution"][0]
            data["coarse_w1"] = data["hw1_i"][1] // self.config["resolution"][0]

        if data["hw0_i"] == data["hw1_i"]:
            ret_dict = self.backbone(
                torch.cat([data["image0_rgb"], data["image1_rgb"]], dim=0),
                (data["coarse_h0"], data["coarse_w0"]),
            )
            feats_c = ret_dict["feats_c"]
            cop_feat = ret_dict["dino_feat"]
            data.update(
                {
                    "feats_x2": ret_dict["feats_x2"],
                    "feats_x1": ret_dict["feats_x1"],
                }
            )
            (feat_c0, feat_c1) = feats_c.split(data["bs"])
            cop_feat0, cop_feat1 = cop_feat.split(data["bs"])
        else:
            ret_dict0 = self.backbone(
                data["image0_rgb"], (data["coarse_h0"], data["coarse_w0"])
            )
            ret_dict1 = self.backbone(
                data["image1_rgb"], (data["coarse_h1"], data["coarse_w1"])
            )
            feat_c0 = ret_dict0["feats_c"]
            feat_c1 = ret_dict1["feats_c"]
            cop_feat0 = ret_dict0["dino_feat"]
            cop_feat1 = ret_dict1["dino_feat"]
            data.update(
                {
                    "feats_x2_0": ret_dict0["feats_x2"],
                    "feats_x1_0": ret_dict0["feats_x1"],
                    "feats_x2_1": ret_dict1["feats_x2"],
                    "feats_x1_1": ret_dict1["feats_x1"],
                }
            )
        if self.get_SPA:
            spec_mask0, spec_mask1 = self.get_SPAmask(data, cop_feat0, cop_feat1)
        data.update(
            {
                "cop_feat0": deepcopy(cop_feat0),
                "cop_feat1": deepcopy(cop_feat1),
            }
        )
        mul = self.config["resolution"][0] // self.config["resolution"][1]
        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul],
                "hw1_f": [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
            }
        )

        mask_c0 = mask_c1 = None
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        data.update(
            {
                "feat_c0_coarse": deepcopy(feat_c0),
                "feat_c1_coarse": deepcopy(feat_c1),
            }
        )
        data["coarse_h0"] = feat_c0.shape[2]
        data["coarse_w0"] = feat_c0.shape[3]
        data["coarse_h1"] = feat_c1.shape[2]
        data["coarse_w1"] = feat_c1.shape[3]
        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")

        if self.config["replace_nan"] and (
            torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))
        ):
            detect_NaN(feat_c0, feat_c1)

        match_logits0 = self.classifier(feat_c0).squeeze(-1)
        match_logits1 = self.classifier(feat_c1).squeeze(-1)

        data.update(
            {
                "match_logits0": match_logits0,
                "match_logits1": match_logits1,
            }
        )
        match_logits0 = torch.sigmoid(match_logits0)
        match_logits1 = torch.sigmoid(match_logits1)
        match_logits0_map = rearrange(
            match_logits0, "n (h w) -> n 1 h w", h=data["coarse_h0"]
        )
        match_logits1_map = rearrange(
            match_logits1, "n (h w) -> n 1 h w", h=data["coarse_h1"]
        )
        up_match_logits0_map = torch.nn.functional.interpolate(
            match_logits0_map,
            (data["hw0_i"][0], data["hw0_i"][1]),
            mode="bilinear",
            align_corners=True,
        )
        up_match_logits1_map = torch.nn.functional.interpolate(
            match_logits1_map,
            (data["hw1_i"][0], data["hw1_i"][1]),
            mode="bilinear",
            align_corners=True,
        )
        data.update(
            {
                "recon_matchable0": up_match_logits0_map,
                "recon_matchable1": up_match_logits1_map,
            }
        )
        return (
            spec_mask0,
            spec_mask1,
            up_match_logits0_map,
            up_match_logits1_map,
            data,
        )

__all__ = ["COP"]
