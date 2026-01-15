# SGAT: Singularity-enhanced Graph Attention Network

**Learning Feature Matching with Singularity-enhanced Graph Attention** (AAAI 2026)

## Overview

SGAT is a novel sparse feature matching method designed to enhance attention to singular points in images during feature interaction. While attention-based approaches have achieved remarkable progress in image feature matching, they still face significant limitations in complex scenarios such as low-texture regions or occlusions.

SGAT addresses these limitations by:
- Leveraging **Co-potentiality** and **Multi-Scale Singularity** as prior guidance
- Designing a **Singularity-aware Attention** mechanism to enhance perception of salient regions
- Developing a **Co-potentiality Guided Attention** mechanism to improve matching potential during feature interaction


## Performance

SGAT achieves state-of-the-art performance on challenging sparse matching benchmarks, especially excelling in:
- Low-texture environments
- Occluded scenes
- Large viewpoint changes
- Illumination variations

## Directory Structure

```
SGAT/
├── paper/                 # Paper PDF and supplementary materials
├── src/
│   ├── backbones/         # Feature extraction backbones (DINO, etc.)
│   ├── extractors/        # Keypoint detectors (SuperPoint, etc.)
│   ├── matchers/          # Matching models
│   │   ├── sgat.py        # Main SGAT matcher
│   │   └── cop/           # Co-potentiality module
│   ├── utils/             # Utility functions
│   ├── geometry/          # Geometry operations
│   ├── base_model.py      # Base model class
│   └── two_view_pipeline.py
├── configs/               # Configuration files
└── README.md
```

## Citation

If you use SGAT in your research, please cite:

```bibtex
@article{sgat2026,
  title={Learning Feature Matching with Singularity-enhanced Graph Attention},
  author={Yizhuo Zhang, Kun Sun, Chang Tang, Yuanyuan Liu, Xin Li},
  journal={AAAI},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

感谢以下开源项目提供的代码和数据支持：
- [LightGlue](https://github.com/cvg/LightGlue) - Transformer-based matching framework
- [DINO](https://github.com/facebookresearch/dino) - Self-supervised vision transformers
- [SuperPoint](https://github.com/rpautrat/SuperPoint) - Self-supervised keypoint detection
- [glue-factory](https://github.com/cvg/glue-factory) - End-to-end feature matching evaluation framework
