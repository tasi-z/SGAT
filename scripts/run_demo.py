from pathlib import Path
import json
import sys

import matplotlib
import numpy as np
import torch
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gluefactory.eval.io import extract_benchmark_conf, load_model
from gluefactory.eval.scannet1500 import Scannet1500Pipeline
from gluefactory.eval.utils import eval_matches_epipolar, eval_relative_pose_robust
from gluefactory.models.cache_loader import CacheLoader
from gluefactory.utils.export_predictions import export_predictions


def image_to_numpy(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def plot_matches(image0, image1, kpts0, kpts1, matches0, scores0, out_path: Path) -> None:
    valid = matches0 > -1
    idx0 = np.where(valid)[0]
    idx1 = matches0[valid]
    score = scores0[valid]
    order = np.argsort(score)[::-1][:100]
    idx0 = idx0[order]
    idx1 = idx1[order]
    score = score[order]

    img0 = image_to_numpy(image0)
    img1 = image_to_numpy(image1)
    h = max(img0.shape[0], img1.shape[0])
    w0, w1 = img0.shape[1], img1.shape[1]

    canvas = np.ones((h, w0 + w1, 3), dtype=np.float32)
    canvas[: img0.shape[0], :w0] = img0
    canvas[: img1.shape[0], w0 : w0 + w1] = img1

    p0 = kpts0[idx0]
    p1 = kpts1[idx1].copy()
    p1[:, 0] += w0

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(canvas)
    ax.scatter(p0[:, 0], p0[:, 1], s=8, c=score, cmap="viridis")
    ax.scatter(p1[:, 0], p1[:, 1], s=8, c=score, cmap="viridis")

    segments = np.stack([p0, p1], axis=1)
    lc = LineCollection(segments, cmap="viridis", linewidths=0.7)
    lc.set_array(score)
    ax.add_collection(lc)
    ax.set_axis_off()
    ax.set_title("Top-100 SGAT Matches on Demo Pair")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "outputs/demo_scannet1500"
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_conf = OmegaConf.load(ROOT / "configs/superpoint+SGAT.yaml")
    conf = OmegaConf.create(Scannet1500Pipeline.default_conf)
    conf = OmegaConf.merge(conf, extract_benchmark_conf(custom_conf, "scannet1500"))
    conf.data.pairs = "demo_scannet1500/pairs.txt"

    pipeline = Scannet1500Pipeline(conf)
    loader = pipeline.get_dataloader(conf.data)
    model = load_model(conf.model, None)
    pred_path = out_dir / "demo_predictions.h5"
    export_predictions(
        loader,
        model,
        pred_path,
        keys=pipeline.export_keys,
        optional_keys=pipeline.optional_export_keys,
    )

    batch = next(iter(pipeline.get_dataloader(conf.data)))
    pred = CacheLoader({"path": str(pred_path), "collate": None}).eval()(batch)

    epi = eval_matches_epipolar(batch, pred)
    pose = eval_relative_pose_robust(batch, pred, conf.eval)

    plot_matches(
        batch["view0"]["image"][0],
        batch["view1"]["image"][0],
        pred["keypoints0"].numpy(),
        pred["keypoints1"].numpy(),
        pred["matches0"].numpy(),
        pred["matching_scores0"].numpy(),
        out_dir / "demo_matches.png",
    )

    print(f"Saved visualization to {out_dir / 'demo_matches.png'}")


if __name__ == "__main__":
    main()
