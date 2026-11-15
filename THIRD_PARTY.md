# Third-party software and models

| Component | Source | License / attribution |
|---|---|---|
| ProtoGCN backbone and recognition/data framework | https://github.com/firework8/ProtoGCN | MIT; original notice in `protogcn/LICENSE` |
| PYSKL, including ST-GCN++ and CTR-GCN | https://github.com/kennymckormick/pyskl | Apache-2.0; original notice in `third_party/pyskl/LICENSE` |
| Skeleton-Cache descriptor/cache port | https://github.com/Alchemist0754/Skeleton-Cache | BSD-3-Clause; `methods/SkeletonCache.LICENSE` |
| ZeroSiam reference | https://github.com/Cascol-Chen/ZeroSiam | Reference implementation identified in `methods/ZeroSiam.py`; closed-set adapter included here |
| NEO reference | https://github.com/awesomealex1/NEO | Reference implementation identified in `methods/NEO.py` |

The included PYSKL source was taken from revision `f2bf3a6b08e2e8dec744692d64efdb187fd6719a`. Original copyright headers and license notices are retained. ProtoGCN includes local support for memory-mapped skeleton annotations and the evaluation pipelines used in this benchmark.

Other methods are the project evaluation adapters for Source, BN, Tent, ETA, EATA, CoTTA, RoTTA, RDumb, and DeYO. Their method-specific settings are retained in `configs/method/`; skeleton augmentations are implemented in those adapters. Comparisons should use the provided adaptations and settings.

ProtoGCN pretrained models originate from its model zoo. ST-GCN++ and CTR-GCN 3D Joint X-View weights originate from the PYSKL model zoo, hosted by OpenMMLab. `checkpoints_manifest.json` records the exact original file digests and the corresponding release files. Evaluation weights retain every state-dictionary tensor; training metadata and optimizer states are removed for the release. Original checkpoint filenames are retained as configuration identifiers and do not imply a new training run.
