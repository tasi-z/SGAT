import warnings
import copy
from pathlib import Path
from typing import Callable, List, Optional
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from ..base_model import BaseModel
from .dino_loftr import full_default_cfg, opt_default_cfg, reparameter

from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics
from ..utils.util import draw_torch_image,get_kpts_mask,draw_kpt_intensity,process_tensor
from ...geometry.gt_generation import (
    gt_line_matches_from_pose_depth,
    gt_matches_from_pose_depth,
)
from .dino_matchable_vis import DINO_matchable
# FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")
FLASH_AVAILABLE = False

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None,weighted_mask : Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            #sim.shape torch.Size([12, 4, 2048, 2048])
            # sim=weighted_mask*sim
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            # attn = F.softmax(sim, -1)
            if weighted_mask is not None:
                attn= F.softmax(sim+weighted_mask, -1)
            else:
                attn= F.softmax(sim, -1)

            # if weighted_mask is not None:
            #     attn= F.softmax(attn*weighted_mask, -1)
            return torch.einsum("...ij,...jd->...id", attn, v),attn

class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.weight_scale = nn.Parameter(torch.ones(1))
        # self.mask_weight_generator = nn.Linear(1, 1)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weighted_mask: Optional[torch.Tensor] = None,
        attention_data: Optional[dict] = None
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        if weighted_mask is not None:
            # weighted_mask torch.Size([2, 1, 2048])
            weighted_mask = weighted_mask.unsqueeze(-2)
            weighted_mask=weighted_mask.expand(-1, self.num_heads, q.shape[-2],-1)
            mask=weighted_mask>0.01
            learnable_weighted_mask=weighted_mask * self.weight_scale
            # learnable_weighted_mask = self.mask_weight_generator(weighted_mask.unsqueeze(-1)).squeeze(-1) # 输入维度
            # weighted_mask torch.Size([2, 4, 2048,2048])
            # q  torch.Size([2, 4, 2048, 64])
        # TODO: 可视化加权前后的注意力变化情况
            # context = self.inner_attn(q, k, v, mask=mask,weighted_mask=weighted_mask)
            context,sim = self.inner_attn(q, k, v, mask=mask,weighted_mask=learnable_weighted_mask)
        else:
            context,sim = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1)),sim


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None,cross_mask0: Optional[torch.Tensor] = None,cross_mask1: Optional[torch.Tensor] = None,attention_data: Optional[dict] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            # sim.shape=torch.Size([12, 4, 2048, 2048])
            if cross_mask0 is not None:
                with torch.no_grad():
                    # cross_mask0：torch.Size([12, 1, 2048]) 展示了qk0中的每个点的可匹配性，越高qk1越应该关注
                    # cross_mask1：torch.Size([12, 1, 2048]) 展示了qk1中的每个点的可匹配性，越高qk0越应该关注
                    cross_mask0 = cross_mask0.unsqueeze(-2) # [batch_size, 1, 1, seq_len_q0]
                    cross_mask1 = cross_mask1.unsqueeze(-2) # [batch_size, 1, 1, seq_len_k1]

                    # 扩展 cross_mask 以匹配相似度矩阵 sim 的形状 [batch_size, heads, seq_len_q0, seq_len_k1]
                    cross_mask0_expanded = cross_mask0.expand(-1, self.heads, qk1.shape[-2], -1) # 沿着 head 维度扩展
                    cross_mask1_expanded = cross_mask1.expand(-1, self.heads, qk0.shape[-2], -1) # 沿着 head 维度扩展, 并转置以匹配方向

                    alpha_div=0.01
                    cross_mask0_binary_bad = cross_mask0_expanded<alpha_div
                    cross_mask1_binary_bad = cross_mask1_expanded<alpha_div
            
                # attn01代表0中的所有点从1中选择性的汇聚权重(b,head,源点数,待选择点数量)
                # 将0中不可匹配点和1中最后一个垃圾桶点相似性置为正无穷
                # 将1中不可匹配点的相似性置为负无穷
                weighted_sim_01=sim
                weighted_sim_01=weighted_sim_01.masked_fill(cross_mask1_binary_bad, -float("inf"))
                # inf_mask_01 = (cross_mask0<alpha_div).transpose(-2, -1).expand(-1, self.heads, -1, -1).squeeze(-1) #0中不可匹配点的mask
                # # weighted_sim_01[:,:,:,-1].masked_fill_(inf_mask_01, weighted_sim_01.max())
                # mask = torch.zeros_like(weighted_sim_01, dtype=torch.bool)
                # mask[:,:,:,-1] = inf_mask_01
                # weighted_sim_01 = torch.where(mask, weighted_sim_01.max(), weighted_sim_01)
                # 应用 cross_mask 进行加权
                weighted_sim_10 = sim.transpose(-2, -1) # 注意转置 sim 以匹配方向
                weighted_sim_10=weighted_sim_10.masked_fill(cross_mask0_binary_bad, -float("inf"))
                # inf_mask_10 = (cross_mask1<alpha_div).transpose(-2, -1).expand(-1, self.heads, -1, -1).squeeze(-1) #0中不可匹配点的mask
                # # weighted_sim_10[:,:,:,-1].masked_fill_(inf_mask_10, weighted_sim_01.max())
                # mask = torch.zeros_like(weighted_sim_10, dtype=torch.bool)
                # mask[:,:,:,-1] = inf_mask_10
                # weighted_sim_10 = torch.where(mask, weighted_sim_10.max(), weighted_sim_10)
                
                attn01 = F.softmax(weighted_sim_01, dim=-1)
                attn10 = F.softmax(weighted_sim_10.contiguous(), dim=-1)
            else:
                attn01 = F.softmax(sim, dim=-1)
                attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            # attn01 = F.softmax(sim, dim=-1)
            # attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            # attn01代表0中的所有点从1中选择性的汇聚权重
            # attn10代表1中的所有点从0中选择性的汇聚权重
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1,attn01,attn10



class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
        weighted_mask0: Optional[torch.Tensor] = None,
        weighted_mask1: Optional[torch.Tensor] = None,
    ):
        
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            attention_data={}
            if weighted_mask0 is not None and weighted_mask1 is not None:
                desc0,self_sim0 = self.self_attn(desc0, encoding0,weighted_mask=weighted_mask0)
                desc1,self_sim1 = self.self_attn(desc1, encoding1,weighted_mask=weighted_mask1)
                # desc0,self_sim0 = self.self_attn(desc0, encoding0)
                # desc1,self_sim1 = self.self_attn(desc1, encoding1)
            else:
                desc0,self_sim0 = self.self_attn(desc0, encoding0)
                desc1,self_sim1 = self.self_attn(desc1, encoding1)
            if weighted_mask0 is not None and weighted_mask1 is not None:
                desc0,desc1,cross_sim0,cross_sim1=self.cross_attn(desc0, desc1,cross_mask0=weighted_mask0,cross_mask1=weighted_mask1)
                # desc0,desc1,cross_sim0,cross_sim1=self.cross_attn(desc0, desc1)
            else:
                desc0,desc1,cross_sim0,cross_sim1=self.cross_attn(desc0, desc1)
            # attention_data={
            #     "self_sim0":self_sim0,
            #     "self_sim1":self_sim1,
            #     "cross_sim0":cross_sim0,
            #     "cross_sim1":cross_sim1
            # }
            return desc0,desc1,attention_data

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    # gather用于根据索引从输入张量中检索值
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1



class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]
    path_denospecmask_dir="/root/autodl-tmp/glue_factory_enc/data/save_dinospec"

    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )


        state_dict = None
        if conf.weights is not None:
            state_dict = torch.load(self.conf.weights, map_location="cuda")
            if "state_dict" in state_dict.keys():
                state_dict = state_dict["state_dict"]
            for k in list(state_dict.keys()):
                if k.startswith("superpoint."):
                    state_dict.pop(k)
                if k.startswith("model."):
                    state_dict[k.replace("model.", "", 1)] = state_dict.pop(k)
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            # self.load_state_dict(state_dict,strict=False)
            self.load_state_dict(state_dict)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
            
        kpts0, kpts1 = data["norm_keypoints0"], data["norm_keypoints1"]
        
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        

        
        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()
        
        
        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)
        
        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training


        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        spec_layer_num=3
        spec_kpt_mask0=data["spec_kpt_mask0"]
        spec_kpt_mask1=data["spec_kpt_mask1"]
        
        all_desc0, all_desc1 = [], []
        self_sim0_list, self_sim1_list, cross_sim0_list, cross_sim1_list = [], [], [], []
        for i in range(self.conf.n_layers):
            if i%2==0:
                if self.conf.checkpointed and self.training:
                    desc0, desc1,attention_data = checkpoint(
                        self.transformers[i], desc0, desc1, encoding0, encoding1,None,None,spec_kpt_mask0,spec_kpt_mask1
                    )
                else:
                    desc0, desc1,attention_data = self.transformers[i](desc0, desc1, encoding0, encoding1,weighted_mask0=spec_kpt_mask0,weighted_mask1=spec_kpt_mask1)
            else:
                if self.conf.checkpointed and self.training:
                    desc0, desc1,attention_data = checkpoint(
                        self.transformers[i], desc0, desc1, encoding0, encoding1
                    )
                else:
                    desc0, desc1,attention_data = self.transformers[i](desc0, desc1, encoding0, encoding1)
            # self_sim0_list.append(attention_data.get("self_sim0",None))
            # self_sim1_list.append(attention_data.get("self_sim1",None))
            # cross_sim0_list.append(attention_data.get("cross_sim0",None))
            # cross_sim1_list.append(attention_data.get("cross_sim1",None))
            # all_desc0.append(desc0)
            # all_desc1.append(desc1)
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0[..., :m-1, :])
                all_desc1.append(desc1[..., :n-1, :])
                continue  # no early stopping or adaptive width at last layer

            # only for eval
            if do_early_stop:
                assert b == 1
                token0, token1 = self.token_confidence[i](desc0, desc1)
                if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
                    break
            if do_point_pruning:
                assert b == 1
                scores0 = self.log_assignment[i].get_matchability(desc0)
                prunemask0 = self.get_pruning_mask(token0, scores0, i)
                keep0 = torch.where(prunemask0)[1]
                ind0 = ind0.index_select(1, keep0)
                desc0 = desc0.index_select(1, keep0)
                encoding0 = encoding0.index_select(-2, keep0)
                prune0[:, ind0] += 1
                scores1 = self.log_assignment[i].get_matchability(desc1)
                prunemask1 = self.get_pruning_mask(token1, scores1, i)
                keep1 = torch.where(prunemask1)[1]
                ind1 = ind1.index_select(1, keep1)
                desc1 = desc1.index_select(1, keep1)
                encoding1 = encoding1.index_select(-2, keep1)
                prune1[:, ind1] += 1

        # desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        desc0, desc1 = desc0[..., :m-1, :], desc1[..., :n-1, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            if do_point_pruning:
                m_indices_0 = ind0[k, m_indices_0]
                m_indices_1 = ind1[k, m_indices_1]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])
        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
            # "matches": matches,
            # "scores": mscores,
            # "self_sim0": self_sim0_list,
            # "self_sim1": self_sim1_list,
            # "cross_sim0": cross_sim0_list,
            # "cross_sim1": cross_sim1_list,
        }

        return pred

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

    def loss(self,pred, data):
        raise NotImplementedError




class SPAGlue(BaseModel):
    default_conf = {"features": "superpoint", **LightGlue.default_conf}
    required_data_keys = [
        "view0",
        "keypoints0",
        "descriptors0",
        "view1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        dconf = OmegaConf.to_container(conf)
        dino_config=full_default_cfg
        dino_config["weights"]="weights/dino_matchable/last.ckpt"
        self.dino_backbone=DINO_matchable(config=dino_config)
        self.dino_backbone.requires_grad_(False)
        self.dino_backbone.eval()
        # 自文件中设置了
        # self.dino_backbone.to(torch.float16)
        # dconf["weights"]="weights/gim/gim_lightglue_100h.ckpt"
        self.lightglue = LightGlue(dconf)
        # for i in range(3,9):
        #     for name, param in self.lightglue.transformers[i].named_parameters():
        #         param.requires_grad = False
        self.loss_fn = NLLLoss(conf.loss)

        self.set_initialized()

    def _forward(self, data):
        

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        
        # 添加kpts0和kpts1各添加一个垃圾箱点并合并到原始点集中的最后
        d = kpts0.shape[-1]  # 特征点的维度
        dummy_kpt0 = torch.zeros(b, 1, d, device=device)  # [b,1,d]
        dummy_kpt1 = torch.zeros(b, 1, d, device=device)  # [b,1,d]
        kpts0 = torch.cat([kpts0, dummy_kpt0], dim=1)  # [b,m+1,d]
        kpts1 = torch.cat([kpts1, dummy_kpt1], dim=1)  # [b,n+1,d]
        data["keypoints0"] = kpts0
        data["keypoints1"] = kpts1
        # 将描述子与垃圾箱点描述子在第1维(点数维度)拼接
        desc0, desc1 = data["descriptors0"], data["descriptors1"]
        d=desc0.shape[-1]
        desc0 = torch.cat([desc0, torch.zeros(b, 1, d, device=device)], dim=1)
        desc1 = torch.cat([desc1, torch.zeros(b, 1, d, device=device)], dim=1)
        data["descriptors0"] = desc0
        data["descriptors1"] = desc1
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()
        data["norm_keypoints0"]=kpts0
        data["norm_keypoints1"]=kpts1
        self.dino_backbone.eval()
        with torch.no_grad():
            dino_data= self.dino_backbone(data["view0"],data["view1"])
            spec_mask0_all,spec_mask1_all=dino_data["spec_mask0"],dino_data["spec_mask1"]
            match_logits0_map,match_logits1_map=dino_data["recon_matchable0"],dino_data["recon_matchable1"]
            data.update(dino_data)
            spec_mask0=process_tensor(match_logits0_map,alpha=0.01,norm=True)
            spec_mask1=process_tensor(match_logits1_map,alpha=0.01,norm=True)
            # spec_mask0=match_logits0_map
            # spec_mask1=match_logits1_map
            # spec_mask0=spec_mask0_all*match_logits0_map
            # spec_mask1=spec_mask1_all*match_logits1_map
            spec_kpt_mask0,spec_kpt_mask1= get_kpts_mask(spec_mask0,data["keypoints0"]),get_kpts_mask(spec_mask1,data["keypoints1"])
        data["spec_kpt_mask0"]=spec_kpt_mask0
        data["spec_kpt_mask1"]=spec_kpt_mask1
        
        # with torch.no_grad():
        res=self.lightglue(data)
        # res["spec_kpt_mask0"]=spec_kpt_mask0
        # res["spec_kpt_mask1"]=spec_kpt_mask1
        # res["spec_mask0"]=spec_mask0
        # res["spec_mask1"]=spec_mask1
        # res["spec_mask0_all"]=spec_mask0_all
        # res["spec_mask1_all"]=spec_mask1_all
        # res["match_logits0_map"]=match_logits0_map
        # res["match_logits1_map"]=match_logits1_map
        # res["spec_kpt_mask0"]=data["spec_kpt_mask0"]
        # res["spec_kpt_mask1"]=data["spec_kpt_mask1"]
        # res["specall_kpt_mask0"]=get_kpts_mask(spec_mask0_all,data["keypoints0"])
        # res["specall_kpt_mask1"]=get_kpts_mask(spec_mask1_all,data["keypoints1"])
        # res["match_kpt_mask0"]=get_kpts_mask(match_logits0_map,data["keypoints0"])
        # res["match_kpt_mask1"]=get_kpts_mask(match_logits1_map,data["keypoints1"])
        return res
    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.lightglue.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        # B = pred['log_assignment'].shape[0]
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.lightglue.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
            ) / (N - 1)

            del params_i
        losses["total"] /= sum_weights

        # confidences
        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics
__main_model__ = SPAGlue
