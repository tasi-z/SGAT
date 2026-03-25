# SGAT: Singularity-enhanced Graph Attention Network

**Learning Feature Matching with Singularity-enhanced Graph Attention**  
AAAI 2026

## Overview

SGAT is a sparse feature matching method for challenging visual correspondence. It strengthens feature interaction around singular and potentially matchable regions, improving robustness in low-texture, occluded, and large-viewpoint scenes.

This release provides a compact evaluation-oriented codebase with:

- The SGAT matcher and minimal runtime dependencies
- A demo image pair with visualization output
- Evaluation entry points for `Scannet1500` and `Scannetv2`

## Highlights

- Singularity-aware attention for emphasizing salient local structures
- Co-potentiality guided interaction for stronger match reasoning
- Sparse matching pipeline built on SuperPoint + SGAT

## Repository Layout

```text
sgat/
├── configs/                  # Evaluation configs
├── data/
│   └── demo_scannet1500/     # Demo pair and pose metadata
├── gluefactory/              # Minimal evaluation and model runtime
├── gluefactory_nonfree/      # SuperPoint extractor
├── scripts/                  # Demo and benchmark entry points
└── weights/                  # Expected local weight layout
```

## Weights

This repository does not commit pretrained binaries. Place the required files under `weights/`:

- `weights/public/sgat_matcher_public.pth` (`sgat.*` only)
- `weights/public/superpoint_extractor_public.pth`
- `weights/cop/last.ckpt` (`COP` backbone weights)
- `weights/dinov2/`

## Installation

Install dependencies with `uv`:

```bash
cd sgat
uv sync
```

Optional local path overrides:

```bash
export SGAT_DATA_PATH=/path/to/datasets
export SGAT_WEIGHTS_PATH=/path/to/weights
export SGAT_EXP_ROOT=/path/to/output_root
```

## Demo

Run the bundled `Scannet1500` example pair:

```bash
cd sgat
uv run python scripts/run_demo.py
```

This writes:

- `outputs/demo_scannet1500/demo_matches.png`

Demo visualization:

![SGAT demo matches](outputs/demo_scannet1500/demo_matches.png)

## Evaluation

Evaluate on `Scannet1500`:

```bash
cd sgat
uv run python scripts/run_scannet1500.py --overwrite
```

Evaluate on `Scannetv2`:

```bash
cd sgat
uv run python scripts/run_scannetv2.py --overwrite
```

## Citation

```bibtex
@article{sgat2026,
  title={Learning Feature Matching with Singularity-enhanced Graph Attention},
  author={Yizhuo Zhang, Kun Sun, Chang Tang, Yuanyuan Liu, Xin Li},
  year={2026},
  volume={40}, 
  number={15}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  month={Mar.}, 
  pages={12952-12960}
}
```

## Acknowledgments

- [LightGlue](https://github.com/cvg/LightGlue)
- [DINO](https://github.com/facebookresearch/dino)
- [SuperPoint](https://github.com/rpautrat/SuperPoint)
- [glue-factory](https://github.com/cvg/glue-factory)