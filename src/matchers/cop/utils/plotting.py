import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import torch

def apply_pca_hw(features_list):
    # 假设特征图的形状为 (channels, height, width)
    total_features = []
    original_shapes = []
    for features in features_list:
        shape = features.shape
        original_shapes.append(shape)
        # 将每个特征图展平成一个向量
        features=features.permute(1,2,0).reshape(-1,shape[0])
        flattened_features = features.view(-1,shape[0]).cpu().numpy()
        total_features.append(flattened_features)

    # 将所有特征向量堆叠在一起进行PCA
    total_features_stacked = np.vstack(total_features)
    n_components=3
    pca = PCA(n_components=n_components)  # 降到1个主成分

    pca.fit(total_features_stacked)
    pca_features_stacked = pca.transform(total_features_stacked)
    for i in range(n_components):
        pca_features_stacked[:, i] = (pca_features_stacked[:, i] - pca_features_stacked[:, i].min()) / \
                            (pca_features_stacked[:, i].max() - pca_features_stacked[:, i].min())
    features_pca_img_list = []
    # # 将降维后的特征重新分配回原始的特征图列表
    start_index = 0
    for i, shape in enumerate(original_shapes):
        num_pixels = np.prod(shape[1:])
        end_index = start_index + num_pixels
        pca_features_current = pca_features_stacked[start_index:end_index]
        # 将降维后的特征值填充回原始形状 (3, height, width)
        pca_features_reshaped = pca_features_current.transpose((1,0)).reshape(n_components, shape[1], shape[2])
        features_pca_img_list.append(pca_features_reshaped)
        start_index = end_index
    return features_pca_img_list

def draw_pca_feature01(feat_c0,feat_c1,title="coarse"):
    features_pca_img_list=apply_pca_hw([feat_c0, feat_c1])
    pca_img_num=len(features_pca_img_list)
    fig, axes = plt.subplots(1, pca_img_num, figsize=(10, 6))
    for i, features_pca_img in enumerate(features_pca_img_list):
        # axes[i].imshow(features_pca_img[0], cmap='viridis')
        # (3,h,w)转换为图像RGB格式
        features_pca_img= np.transpose(features_pca_img, (1, 2, 0))
        axes[i].imshow(features_pca_img)
        axes[i].axis('off')
        axes[i].set_title(f'PCA of {title}{i}')

    plt.tight_layout()
    # plt.show()
    return fig
def draw_recon_feature_img(batch,key0='recon_feature_img0',key1='recon_feature_img1',ori_key0='image0',ori_key1='image1'):

    # 假设你有四个图像数据
    # .squeeze()用于去除维度为1的维度
    image0 = batch[ori_key0][0].squeeze().cpu().numpy()
    recon_feature_img0 = batch[key0][0].squeeze().cpu().numpy()
    image1 = batch[ori_key1][0].squeeze().cpu().numpy()
    recon_feature_img1 = batch[key1][0].squeeze().cpu().numpy()

    # 创建一个2行2列的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 在第一个子图中展示image0
    axes[0, 0].imshow(image0, cmap='gray')
    axes[0, 0].axis('off')  # 关闭第一个子图的坐标轴
    axes[0, 0].set_title(f'{ori_key0}')  # 设置第一个子图的标题

    # 在第二个子图中展示recon_feature_img0
    axes[0, 1].imshow(recon_feature_img0, cmap='gray')
    axes[0, 1].axis('off')  # 关闭第二个子图的坐标轴
    axes[0, 1].set_title(f'{key0}')  # 设置第二个子图的标题

    # 在第三个子图中展示image1
    axes[1, 0].imshow(image1, cmap='gray')
    axes[1, 0].axis('off')  # 关闭第三个子图的坐标轴
    axes[1, 0].set_title(f'{ori_key1}')  # 设置第三个子图的标题

    # 在第四个子图中展示recon_feature_img1
    axes[1, 1].imshow(recon_feature_img1, cmap='gray')
    axes[1, 1].axis('off')  # 关闭第四个子图的坐标轴
    axes[1, 1].set_title(f'{key1}')  # 设置第四个子图的标题

    # 调整子图之间的间距
    plt.tight_layout()
    return fig
def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    # txt_color = 'k' if img0[:100, :200].mean() > 200 else 'darkorange'
    brightness = img0[:100, :200].mean()
    # 根据亮度选择文本颜色
    if brightness > 200:
        txt_color = 'black'  # 如果背景非常亮，使用黑色
    elif brightness > 150:
        txt_color = 'cyan'  # 如果背景较亮，使用青色
    elif brightness > 100:
        txt_color = 'magenta'  # 如果背景中等亮度，使用洋红色
    else:
        txt_color = 'yellow'  # 如果背景较暗，使用黄色
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic'):
    b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].cpu().numpy()
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()[[1, 0]]
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()[[1, 0]]

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure

def _make_confidence_figure(data, b_id,with_gt=False):
    # TODO: Implement confidence figure
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    pre_matchable0 = data['recon_matchable0'][b_id][0].cpu().numpy()
    # 将pre_matchable0>0.6的像素点置为1，否则置为0
    # pre_matchable0[pre_matchable0 > 0.6] = 1
    # pre_matchable0[pre_matchable0 <= 0.6] = 0
    pre_matchable1 = data['recon_matchable1'][b_id][0].cpu().numpy()
    # pre_matchable1[pre_matchable1 > 0.6] = 1
    # pre_matchable1[pre_matchable1 <= 0.6] = 0
    if with_gt:
        gt_matchable0 = data['overlap_mask0'][b_id][0].cpu().numpy()
        gt_matchable1 = data['overlap_mask1'][b_id][0].cpu().numpy()
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # 在第一个子图中显示 img0
    axes[0, 0].imshow(img0, cmap='gray')
    axes[0, 0].set_title('Image 0')
    axes[0, 0].axis('off')  # 隐藏坐标轴

    # 在第二个子图中显示 img1
    axes[0, 1].imshow(img1, cmap='gray')
    axes[0, 1].set_title('Image 1')
    axes[0, 1].axis('off')

    # 在第三个子图中显示 pre_matchable0
    im0 = axes[1, 0].imshow(pre_matchable0, cmap='plasma', interpolation='nearest') # 使用 'viridis' 颜色映射
    axes[1, 0].set_title('Pre Matchable 0')
    axes[1, 0].axis('off')
    plt.colorbar(im0, ax=axes[1, 0], shrink=0.7) # 添加色彩条，并缩小尺寸

    # 在第四个子图中显示 pre_matchable1
    im1 = axes[1, 1].imshow(pre_matchable1, cmap='plasma', interpolation='nearest') # 使用 'plasma' 颜色映射
    axes[1, 1].set_title('Pre Matchable 1')
    axes[1, 1].axis('off')
    plt.colorbar(im1, ax=axes[1, 1], shrink=0.7) # 添加色彩条，并缩小尺寸
    if with_gt:
        axes[2, 0].imshow(gt_matchable0*255, cmap='gray', interpolation='nearest')
        axes[2, 0].set_title('GT Matchable 0')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(gt_matchable1*255, cmap='gray', interpolation='nearest')
        axes[2, 1].set_title('GT Matchable 1')
        axes[2, 1].axis('off')
    # 调整子图之间的间距
    plt.tight_layout()
    return fig

def make_matching_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'gt']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)