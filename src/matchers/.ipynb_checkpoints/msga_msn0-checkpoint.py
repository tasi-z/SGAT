import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
import os
import sys
# from ...settings import DATA_PATH
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from gluefactory.settings import DATA_PATH
from gluefactory.models.utils.losses import NLLLoss
from gluefactory.models.utils.metrics import matcher_metrics
from pathlib import Path

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True

def seeding(nn_index1,nn_index2,x1,x2,topk,match_score,confbar,nms_radius,use_mc=True,test=False):
    
    #apply mutual check before nms
    if use_mc:
        mask_not_mutual=nn_index2.gather(dim=-1,index=nn_index1)!=torch.arange(nn_index1.shape[1],device='cuda')
        match_score[mask_not_mutual]=-1
    #NMS
    pos_dismat1=((x1.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x1.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x1@x1.transpose(1,2))).abs_().sqrt_()
    x2=x2.gather(index=nn_index1.unsqueeze(-1).expand(-1,-1,2),dim=1)
    pos_dismat2=((x2.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x2.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x2@x2.transpose(1,2))).abs_().sqrt_()
    radius1, radius2 = nms_radius * pos_dismat1.mean(dim=(1,2),keepdim=True), nms_radius * pos_dismat2.mean(dim=(1,2),keepdim=True)
    nms_mask = (pos_dismat1 >= radius1) & (pos_dismat2 >= radius2)
    mask_not_local_max=(match_score.unsqueeze(-1)>=match_score.unsqueeze(-2))|nms_mask
    mask_not_local_max=~(mask_not_local_max.min(dim=-1).values)
    match_score[mask_not_local_max] = -1
 
    #confidence bar
    match_score[match_score<confbar]=-1
    mask_survive=match_score>0
    if test:
        topk=min(mask_survive.sum(dim=1)[0]+2,topk)
    _,topindex = torch.topk(match_score,topk,dim=-1)#b*k
    seed_index1,seed_index2=topindex,nn_index1.gather(index=topindex,dim=-1)
    return seed_index1,seed_index2

def getseed_from_dist(aug_desc1,aug_desc2,x1,x2,seed_top_k,conf_bar,seed_radius_coe,test_mode=False):
#初始seed
    desc1_nor, desc2_nor = aug_desc1.transpose(1,2), aug_desc2.transpose(1,2)
    # desc1_nor, desc2_nor = aug_desc1, aug_desc2
    desc1_nor, desc2_nor = torch.nn.functional.normalize(desc1_nor,dim=-1), torch.nn.functional.normalize(desc2_nor,dim=-1)
    desc_dismat=(2-2*torch.matmul(desc1_nor,desc2_nor.transpose(1,2))).sqrt_()
    values,nn_index=torch.topk(desc_dismat,k=2,largest=False,dim=-1,sorted=True)
    nn_index2=torch.min(desc_dismat,dim=1).indices.squeeze(1)
    inverse_ratio_score,nn_index1=values[:,:,1]/values[:,:,0],nn_index[:,:,0]#get inverse score
    seed_index1,seed_index2=seeding(nn_index1,nn_index2,x1,x2,seed_top_k,inverse_ratio_score,conf_bar,\
                            seed_radius_coe,test=test_mode) 
    return seed_index1,seed_index2,nn_index1

def get_seed_from_scores(p, x1, x2, seed_top_k, conf_bar, seed_radius_coe, test_mode=False):
    # Rematching with p
    values, nn_index = torch.topk(p[:, :-1, :-1], k=1, dim=-1)
    nn_index2 = torch.max(p[:, :-1, :-1], dim=1).indices.squeeze(1)
    p_match_score, nn_index1 = values[:, :, 0], nn_index[:, :, 0]
    # Reseeding
    seed_index1, seed_index2 = seeding(nn_index1, nn_index2, x1, x2, seed_top_k, p_match_score, conf_bar, seed_radius_coe, test=test_mode)
    return seed_index1, seed_index2

def cal_orth(x,y):
    #计算x和y的正交向量，得到的向量与x正交
    # x=x.permute(0,2,1)
    # y=y.permute(0,2,1)
    res=torch.zeros_like(x)        
    for i in range(x.shape[0]):#对于每个batch
        proj = torch.matmul(x[i], y[i].t())
        x_norm = torch.norm(x[i], p=2, dim=1, keepdim=True)
        y_norm = torch.norm(y[i], p=2, dim=1, keepdim=True)
        proj=torch.diag(proj)#diag是对角线元素
        proj_A=proj/(x_norm**2).t()#
        proj_A=x[i]*proj_A.reshape(-1,1)#y在x上的投影
        vertical_x_y=y[i]-proj_A #正交
        res[i]=vertical_x_y
    # res=res.permute(0,2,1)
    return res

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

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)

#自注意力机制
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
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

#cross-attention,对两个描述子进行交叉注意力(两个都更新)
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
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
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
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class attention_propagantion(nn.Module):
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
    
    def map_1(self, func: Callable, x0: torch.Tensor):
        return func(x0)
    
    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None,weight_v=None
    ) -> List[torch.Tensor]:
        batch_size=x0.shape[0]
        qk0, qk1 = self.map_(self.to_qk, x0, x1)#qk0是x0的query，qk1是x1的key
        v0, v1 = self.map_(self.to_v, x0, x1)#v0是x0的value，v1是x1的value
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)#相似性矩阵
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            if weight_v is not None:
                v1=v1*weight_v.view(batch_size,1,-1,1)
            attn01 = F.softmax(sim, dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            if mask is not None:
                m0 = m0.nan_to_num()
        m0 = self.map_1(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0)
        m0 = self.map_1(self.to_out, m0)
        x0_new = x0 + self.ffn(torch.cat([x0, m0], -1))#m0即add_value,即对x0的更新
        return x0_new        
      
class PointCN(nn.Module):
    def __init__(self, channels,out_channels):
        nn.Module.__init__(self)
        # self.shot_cut = nn.Linear(channels, out_channels)
        self.shot_cut = nn.Linear(channels, out_channels)
        self.conv = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.LayerNorm(channels, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x) + self.shot_cut(x)
    
class SeedBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.attention_block_down = attention_propagantion(embed_dim, num_heads)
        self.cluster_filter=nn.Sequential(nn.Linear(2*embed_dim,2*embed_dim), nn.LayerNorm(2*embed_dim), nn.ReLU(),
                                         nn.Linear(2*embed_dim, 2*embed_dim))
        self.cross_filter=attention_propagantion(embed_dim,num_heads)
        self.confidence_filter=PointCN(2*embed_dim,1)
        self.attention_block_self=attention_propagantion(embed_dim,num_heads)
        self.attention_block_up=attention_propagantion(embed_dim,num_heads)

        # self.cat_filter=nn.Sequential(nn.Linear(2*embed_dim,2*embed_dim), nn.LayerNorm(2*embed_dim), nn.ReLU(),
        #                               nn.Linear(2*embed_dim, embed_dim))

    def forward(
        self,
        desc1,
        desc2,
        seed_index1,
        seed_index2,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
                #抽出种子点
        cluster1, cluster2 = desc1.gather(dim=1, index=seed_index1.unsqueeze(2).expand(-1, -1, self.embed_dim)), \
                             desc2.gather(dim=1, index=seed_index2.unsqueeze(2).expand(-1, -1, self.embed_dim))
        
        #pooling
        cluster1, cluster2 = self.attention_block_down(cluster1, desc1), self.attention_block_down(cluster2, desc2)
        concate_cluster=self.cluster_filter(torch.cat([cluster1,cluster2],dim=-1))
        #filtering
        cluster1,cluster2=self.cross_filter(concate_cluster[:,:,:self.embed_dim],concate_cluster[:,:,self.embed_dim:]),\
                        self.cross_filter(concate_cluster[:,:,self.embed_dim:],concate_cluster[:,:,:self.embed_dim])
        cluster1,cluster2=self.attention_block_self(cluster1,cluster1),self.attention_block_self(cluster2,cluster2)
        #unpooling
        seed_weight=self.confidence_filter(torch.cat([cluster1,cluster2],dim=-1))#最后一维是特征维度
        seed_weight=torch.sigmoid(seed_weight).squeeze(-1)#squeeze
        desc1_seed,desc2_seed=self.attention_block_up(desc1,cluster1,weight_v=seed_weight),self.attention_block_up(desc2,cluster2,weight_v=seed_weight)
        return desc1_seed,desc2_seed,seed_weight

class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        embed_dim=args[0]
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)
        self.seed_attn=SeedBlock(*args, **kwargs)
        self.cat_filter = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        seed_index0,
        seed_index1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            #lightglue的全连接层
            desc0_all = self.self_attn(desc0, encoding0)
            desc1_all = self.self_attn(desc1, encoding1)
            desc0_all, desc1_all = self.cross_attn(desc0_all, desc1_all)
            #sgmnet的seed层
            desc0_seed, desc1_seed,seed_weight = self.seed_attn(desc0, desc1, seed_index0, seed_index1)
            #正交融合
            desc0_orth,desc1_orth=cal_orth(desc0_all,desc0_seed),cal_orth(desc1_all,desc1_seed)
            desc0,desc1=desc0+self.cat_filter(torch.cat([desc0_all,desc0_orth],dim=-1)),desc1+self.cat_filter(torch.cat([desc1_all,desc1_orth],dim=-1))
            return desc0, desc1,seed_weight

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        desc0, desc1 = self.cross_attn(desc0, desc1, mask)
        return desc0, desc1


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


class MSGA(nn.Module):
    default_conf = {
        "name": "msga",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "seed_top_k":[200,200],
        "seed_radius_coe":0.01,
        "seedlayer":[0,6],
        "use_mc_seeding":True,
        "use_score_encoding":False,
        "conf_bar":[1.11,0.1],
        "sink_iter":[10,100],
        "detach_iter":150000,
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

        self.loss_fn = NLLLoss(conf.loss)

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:
                fname = (
                    f"{conf.weights}_{conf.weights_from_version}".replace(".", "-")
                    + ".pth"
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    self.url.format(conf.weights_from_version, conf.weights),
                    file_name=fname,
                )

        if state_dict:
            # rename old state dict entries
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

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

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

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
        if self.training:
        #获取种子
            seed_index0,seed_index1,nn_index0=getseed_from_dist(desc0.transpose(1,2),desc1.transpose(1,2),kpts0,kpts1,self.conf.seed_top_k[0],self.conf.conf_bar[0], self.conf.seed_radius_coe,False)
        else:
            seed_index0,seed_index1,nn_index0=getseed_from_dist(desc0.transpose(1,2),desc1.transpose(1,2),kpts0,kpts1,self.conf.seed_top_k[0],self.conf.conf_bar[0], self.conf.seed_radius_coe,True)


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
        do_early_stop = self.conf.depth_confidence > 0 and not self.training# self.training用于判断是否为训练模式,是nn.Module的属性
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []
        seed_weight_tower=[]

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1,seed_weight= checkpoint(
                    self.transformers[i], desc0, desc1, encoding0, encoding1,seed_index0,seed_index1
                )
            else:
                desc0, desc1,seed_weight = self.transformers[i](desc0, desc1, encoding0, encoding1,seed_index0,seed_index1)
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                seed_weight_tower.append(seed_weight)
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

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
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
            "seed_conf":torch.stack(seed_weight_tower, 1)
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

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
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

            losses["confidence"] += self.token_confidence[i].loss(
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
        #seed confidences损失
        
        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = MSGA
