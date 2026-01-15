from __future__ import annotations

import einops as E
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from transformers.models.vit_mae.modeling_vit_mae import (
    get_2d_sincos_pos_embed_from_grid,
)

def draw_patch_match(rgb_0, rgb_1, idx_0to1_consistent, matchable_map0=None, matchable_map1=None, title="patchmatch"):
    if matchable_map0 is not None:
        b, _, target_h0, target_w0 = matchable_map0.shape
        b, _, target_h1, target_w1 = matchable_map1.shape
    else:
        b, _, target_h0, target_w0 = rgb_0.shape
        b, _, target_h1, target_w1 = rgb_1.shape
    
    for i in range(b):
        this_rgb_0 = rgb_0[i].permute(1, 2, 0).detach().cpu().numpy()
        this_rgb_1 = rgb_1[i].permute(1, 2, 0).detach().cpu().numpy()
        
        inp_this_rgb_0 = cv2.resize(this_rgb_0, (target_w0, target_h0))
        inp_this_rgb_1 = cv2.resize(this_rgb_1, (target_w1, target_h1))
        
        this_idx_0to1_consistent = idx_0to1_consistent[i].detach().cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(inp_this_rgb_0/255)
        ax1.set_title('Image 0')
        ax1.axis('off')
        
        ax2.imshow(inp_this_rgb_1/255)
        ax2.set_title('Image 1')
        ax2.axis('off')
        
        h0, w0, _ = inp_this_rgb_0.shape
        h1, w1, _ = inp_this_rgb_1.shape
        
        # 生成随机颜色
        colors = np.random.rand(len(this_idx_0to1_consistent), 3)
        
        # 相同匹配采用同一个颜色,并绘制连接线 
        for idx, color in zip(range(len(this_idx_0to1_consistent)), colors):
            if this_idx_0to1_consistent[idx] == -1:
                continue
            
            y0, x0 = divmod(idx, w0)
            y1, x1 = divmod(this_idx_0to1_consistent[idx], w1)
            
            ax1.scatter([x0], [y0], c=[color], s=10)
            ax2.scatter([x1], [y1], c=[color], s=10)
            
            # 绘制连接线
            con_fig = fig.transFigure.inverted()
            coord1 = con_fig.transform(ax1.transData.transform([x0, y0]))
            coord2 = con_fig.transform(ax2.transData.transform([x1, y1]))
            line = plt.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), 
                               transform=fig.transFigure, color=color, linewidth=0.5, alpha=0.5)
            fig.lines.append(line)
        
        plt.savefig(f"{title}_{i}.png", bbox_inches='tight')
        plt.close(fig)
        
def get_matchable_map(feats0, feats1):
    B, C, H0, W0 = feats0.shape
    _, _, H1, W1 = feats1.shape

    # 将特征向量归一化
    feats0 = feats0.view(B, C, -1)
    feats1 = feats1.view(B, C, -1)
    feats0 = feats0 / (torch.norm(feats0, dim=1, keepdim=True) + 1e-7)
    feats1 = feats1 / (torch.norm(feats1, dim=1, keepdim=True) + 1e-7)

    # 计算相似度矩阵
    sim = torch.bmm(feats0.transpose(1, 2), feats1)

    # 计算每个位置的最大相似度和对应的索引
    max_sim_0to1, idx_0to1 = torch.max(sim, dim=2)
    max_sim_1to0, idx_1to0 = torch.max(sim, dim=1)

    # 创建索引张量
    idx_0 = torch.arange(H0 * W0, device=feats0.device).view(1, -1).expand(B, -1)
    idx_1 = torch.arange(H1 * W1, device=feats1.device).view(1, -1).expand(B, -1)

    # 检查双向匹配
    consistent_0 = idx_1to0.gather(1, idx_0to1) == idx_0
    consistent_1 = idx_0to1.gather(1, idx_1to0) == idx_1
    # 不存在存在双向匹配的索引 置为-1 
    idx_0to1_consistent = idx_0to1.clone()
    idx_0to1_consistent[ ~consistent_0 ] = -1
    # max_sim_0to1<0.6的位置置为-1
    idx_0to1_consistent[ max_sim_0to1<0.9 ] = -1
    # idx_0to1_consistent不为-1的数量
    matchable_num=(idx_0to1_consistent!=-1).sum()
    # 将没有双向匹配的位置置为0
    matchable_map0 = max_sim_0to1 * consistent_0.float()
    matchable_map1 = max_sim_1to0 * consistent_1.float()

    # 将相似度重塑为原始特征图的形状
    matchable_map0 = matchable_map0.view(B, 1, H0, W0)
    matchable_map1 = matchable_map1.view(B, 1, H1, W1)

    return matchable_map0, matchable_map1,idx_0to1_consistent

def draw_kpt_intensity(image, kpts, spec_kpt_mask, title="intensity",save_dir=None):
    b, c, h, w = image.shape
    b, kpt_num, _ = kpts.shape
    b, _, kpt_num = spec_kpt_mask.shape
    # spec_kpt_mask强度越大,颜色越浅
    # spec_kpt_mask>0.9的关键点不绘制
    fig=plt.figure(figsize=(10, 10))
    for i in range(b):
        i_image = image[i].permute(1, 2, 0).detach().cpu().numpy()
        # i_image = cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 10))
        i_kpts = kpts[i].detach().cpu().numpy()
        i_spec_kpt_mask = spec_kpt_mask[i, 0].detach().cpu().numpy()
        plt.imshow(i_image)
        min_val = np.min(i_spec_kpt_mask)
        max_val = np.max(i_spec_kpt_mask)
        normalized_mask = (i_spec_kpt_mask - min_val) / (max_val - min_val)
        color_map = plt.get_cmap('Blues')
        colors = color_map(normalized_mask)
        for j in range(kpt_num):
            x, y = i_kpts[j]
            if i_spec_kpt_mask[j] >= 0.3:
                # 计算颜色：强度越大，半径越小
                # r = 1 - normalized_mask[j]  
                r = normalized_mask[j]  
                # r缩放为0-5
                r= r*10
                plt.scatter(x, y, s=r, color=colors[j], alpha=0.5)
                # plt.plot(x, y, 'o', markersize=3, color=(color, color, color))
            else:
                plt.scatter(x, y, s=2, color=colors[j], alpha=0.5)

        plt.title(title)
        # plt.axis('off')
        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches='tight')
            # plt.savefig(f"{title}_{i}.png", bbox_inches='tight')
        # plt.close()
    return fig
        
def get_kpts_feat(feat_map, kpts, image_size):
    # spec_mask.shape  torch.Size([2, 1, 1024, 1024]) 
    b, c, h, w = feat_map.shape
    # kpts.shape  torch.Size([2, 2048, 2])
    b, kpt_num, _ = kpts.shape
    
    # 计算图像与特征图的缩放比例
    scale_x = (w / image_size[:, 0]).view(b, 1)
    scale_y = (h / image_size[:, 1]).view(b, 1)
    
    # 将关键点坐标从图像坐标转换为特征图坐标
    kpts_x = (kpts[:, :, 0] * scale_x).round().long()
    kpts_y = (kpts[:, :, 1] * scale_y).round().long()
    # 裁剪索引，防止越界
    kpts_x = kpts_x.clamp(0, w - 1)
    kpts_y = kpts_y.clamp(0, h - 1)
    # 创建批次索引
    batch_indices = torch.arange(b).unsqueeze(1).expand(-1, kpt_num)  # [b, kpt_num]

    # 展开索引张量
    batch_indices_flat = batch_indices.reshape(-1)  # [b * kpt_num]
    kpts_y_flat = kpts_y.reshape(-1)               # [b * kpt_num]
    kpts_x_flat = kpts_x.reshape(-1)               # [b * kpt_num]

    # 使用高级索引提取特征
    spec_kpt_feat = feat_map[batch_indices_flat, :, kpts_y_flat, kpts_x_flat]  # [b * kpt_num, c]
    # 调整输出形状
    spec_kpt_feat = spec_kpt_feat.view(b, kpt_num, c)  

    return spec_kpt_feat
        
def process_tensor(tensor,alpha=0.05,norm=False):
    # 创建掩码，将小于0.05的值设为0
    mask = tensor >= alpha
    tensor = tensor * mask
    
    # 获取最小值和最大值
    min_val = tensor.min()
    max_val = tensor.max()
    
    # 归一化到0-1
    if norm:
        if max_val != min_val:        
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else: 
            normalized_tensor = tensor
    else: 
        normalized_tensor = tensor
        
    return normalized_tensor
def get_kpts_mask(spec_mask, kpts):
    # spec_mask.shape  torch.Size([2, 1, 1024, 1024]) 
    # kpts.shape  torch.Size([2, 2048, 2])
    b, kpt_num, _ = kpts.shape
    # 将关键点坐标转换为像素坐标,不应该向上取整,会溢出
    kpts_x = (kpts[:, :, 0]).long()
    kpts_y = (kpts[:, :, 1]).long()
    # kpts_x = (kpts[:, :, 0]).round().long()
    # kpts_y = (kpts[:, :, 1]).round().long()
    # 创建批次索引
    batch_indices = torch.arange(b, device=spec_mask.device).view(b, 1).expand(b, kpt_num)
    # 从spec_mask中抽取关键点位置的值
    spec_kpt_mask0 = spec_mask[batch_indices, 0, kpts_y, kpts_x]
    # 重塑为所需的形状 [2, 1, 2048]
    spec_kpt_mask0 = spec_kpt_mask0.unsqueeze(1)
    return spec_kpt_mask0


def draw_torch_image(image, title="",type="cmap",bg_img=None,save_dir=None):
    b,c,h,w=image.shape
    if c>3:
        image=image[:,0:1,:,:]
    image=image.permute(0,2,3,1)
    fig=plt.figure(figsize=(6, 6))
    if type=="rgb":
        for i in range(b):
            i_image=image[i]
            i_image=i_image.detach().cpu().numpy()
            i_image = (i_image).astype('uint8')
            i_image=cv2.cvtColor(i_image, cv2.COLOR_BGR2RGB)
            plt.imshow(i_image)
            plt.title(title)
            plt.axis('off')
            plt.savefig(f"{title}_{i}.png")
            #关闭
            # plt.close()
    if type=="cmap":
        for i in range(b):
            i_image=image[i]
            # i_image = (i_image + 1) / 2
            if bg_img is not None:
                cur_bg_img=bg_img[i].permute(1,2,0).detach().cpu().numpy()
                plt.imshow(cur_bg_img)
                plt.imshow(i_image.detach().cpu().numpy(), cmap='Blues', interpolation='nearest', alpha=0.5)
            else:
                plt.imshow(i_image.detach().cpu().numpy(), cmap='Blues', interpolation='nearest')
            plt.colorbar()
            plt.title(title)
            if save_dir is not None:
                # plt.savefig(save_dir, bbox_inches='tight')
                plt.savefig(f"{title}_{i}.png")
            #关闭
            # plt.close()
    return fig

def resize_pos_embed(
    pos_embed: torch.Tensor, hw: tuple[int, int], has_cls_token: bool = True
):
    """
    Resize positional embedding for arbitrary image resolution. Resizing is done
    via bicubic interpolation.

    Args:
        pos_embed: Positional embedding tensor of shape ``(n_patches, embed_dim)``.
        hw: Target height and width of the tensor after interpolation.
        has_cls_token: Whether ``pos_embed[0]`` is for the ``[cls]`` token.

    Returns:
        Tensor of shape ``(new_n_patches, embed_dim)`` of resized embedding.
        ``new_n_patches`` is ``new_height * new_width`` if ``has_cls`` is False,
        else ``1 + new_height * new_width``.
    """

    n_grid = pos_embed.shape[0] - 1 if has_cls_token else pos_embed.shape[0]

    # Do not resize if already in same shape.
    if n_grid == hw[0] * hw[1]:
        return pos_embed

    # Get original position embedding and extract ``[cls]`` token.
    if has_cls_token:
        cls_embed, pos_embed = pos_embed[[0]], pos_embed[1:]

    orig_dim = int(pos_embed.shape[0] ** 0.5)

    pos_embed = E.rearrange(pos_embed, "(h w) c -> 1 c h w", h=orig_dim)
    pos_embed = F.interpolate(
        pos_embed, hw, mode="bicubic", align_corners=False, antialias=True
    )
    pos_embed = E.rearrange(pos_embed, "1 c h w -> (h w) c")

    # Add embedding of ``[cls]`` token back after resizing.
    if has_cls_token:
        pos_embed = torch.cat([cls_embed, pos_embed], dim=0)

    return pos_embed


def center_padding(images, patch_size):
    _, _, h, w = images.shape
    diff_h = h % patch_size
    diff_w = w % patch_size

    if diff_h == 0 and diff_w == 0:
        return images

    pad_h = patch_size - diff_h
    pad_w = patch_size - diff_w

    pad_t = pad_h // 2
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_b = pad_h - pad_t

    images = F.pad(images, (pad_l, pad_r, pad_t, pad_b))
    return images

def get_self_similarity(featmap):
    # 将特征图展平 [B, C, H*W]
    B, C, H, W = featmap.shape
    featmap_flat = featmap.view(B, C, -1)  # [B, C, H*W]

    # 计算特征图展平后的范数，并保持维度
    featmap_norm = torch.norm(featmap_flat, dim=1, keepdim=True)  # [B, 1, H*W]

    # 批量点积操作，使用 bmm 进行批量矩阵乘法 [B, H*W, H*W]
    dot_product_matrix = torch.bmm(featmap_flat.transpose(1, 2), featmap_flat)

    # 计算归一化后的余弦相似性，防止除0错误
    similarity_matrix = dot_product_matrix / (featmap_norm.transpose(1, 2) * featmap_norm + 1e-8)

    # 剔除对角线元素
    batch_indices = torch.arange(B).unsqueeze(1).unsqueeze(2)
    diag_indices = torch.arange(H * W)
    similarity_matrix[batch_indices, diag_indices, diag_indices] = float('-inf')

    # 计算最大相似性 [B, H*W]
    max_similarity, _ = similarity_matrix.max(dim=2)
    #归一化到0-1之间
    max_similarity = (max_similarity - max_similarity.min()) / (max_similarity.max() - max_similarity.min())
    # 将结果还原为特征图的形状 [B, H, W]
    max_similarity_map = max_similarity.view(B,1, H, W)

    return max_similarity_map

def get_max_cos_similarity(featmap0, featmap1):
    def compute_similarity(featmap_a, featmap_b):
        B, C, H_a, W_a = featmap_a.shape
        _, _, H_b, W_b = featmap_b.shape

        # 将特征图展平 [B, C, H*W]
        featmap_flat_a = featmap_a.view(B, C, -1)  # [B, C, H_a*W_a]
        featmap_flat_b = featmap_b.view(B, C, -1)  # [B, C, H_b*W_b]

        # 计算特征图展平后的范数，并保持维度
        featmap_norm_a = torch.norm(featmap_flat_a, dim=1, keepdim=True)  # [B, 1, H_a*W_a]
        featmap_norm_b = torch.norm(featmap_flat_b, dim=1, keepdim=True)  # [B, 1, H_b*W_b]

        # 批量点积操作，使用 bmm 进行批量矩阵乘法 [B, H_a*W_a, H_b*W_b]
        dot_product_matrix = torch.bmm(featmap_flat_a.transpose(1, 2), featmap_flat_b)

        # 计算归一化后的余弦相似性，防止除0错误
        similarity_matrix = dot_product_matrix / (featmap_norm_a.transpose(1, 2) * featmap_norm_b + 1e-8)

        # 计算最大相似性 [B, H_a*W_a]
        max_similarity, _ = similarity_matrix.max(dim=2)

        # 归一化到0-1之间
        max_similarity = (max_similarity - max_similarity.min()) / (max_similarity.max() - max_similarity.min())

        # 将结果还原为特征图的形状 [B, H_a, W_a]
        max_similarity_map = max_similarity.view(B, 1, H_a, W_a)

        return max_similarity_map

    # 从 featmap0 到 featmap1 的最大相似性
    max_similarity_map_0_to_1 = compute_similarity(featmap0, featmap1)

    # 从 featmap1 到 featmap0 的最大相似性
    max_similarity_map_1_to_0 = compute_similarity(featmap1, featmap0)

    return max_similarity_map_0_to_1, max_similarity_map_1_to_0

def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    COPIED FROM TRANSFORMERS PACKAGE AND EDITED TO ALLOW FOR DIFFERENT WIDTH-HEIGHT
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or
        (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def tokens_to_output(output_type, dense_tokens, cls_token, feat_hw):
    if output_type == "cls":
        assert cls_token is not None
        output = cls_token
    elif output_type == "gap":
        output = dense_tokens.mean(dim=1)
    elif output_type == "dense":
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        output = dense_tokens.contiguous()
    elif output_type == "dense-cls":
        assert cls_token is not None
        h, w = feat_hw
        dense_tokens = E.rearrange(dense_tokens, "b (h w) c -> b c h w", h=h, w=w)
        cls_token = cls_token[:, :, None, None].repeat(1, 1, h, w)
        output = torch.cat((dense_tokens, cls_token), dim=1).contiguous()
    else:
        raise ValueError()

    return output
