<div align="center">

# When the Sensor Stays Broken<br>Test-Time Robust Skeleton-Based Action Recognition

**SRT · NTU60-C · NTU120-C**

[![Python](https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white)](#-installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch&logoColor=white)](#-installation)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](#-installation)

[![NTU60-C](https://img.shields.io/badge/NTU60--C-X--View-6366F1)](#-benchmarks)
[![NTU120-C](https://img.shields.io/badge/NTU120--C-X--Sub-0D9488)](#-benchmarks)
[![License](https://img.shields.io/badge/License-MIT-64748B)](LICENSE)

Anonymous code and benchmark release.

[📊 Benchmarks](#-benchmarks) · [🛠️ Installation](#-installation) · [🚀 Quick Start](#-quick-start) · [📦 Data & Checkpoints](#-data--checkpoints)

</div>

---

**SRT** adapts skeleton-based action recognizers to persistent sensor failures. This repository provides the [method implementation](methods/SRT_TTA_paper.py), comparison methods, reproducible experiments, and corruption generators.

## 📊 Benchmarks

| Benchmark | Split | Sequences per domain | Corruptions | Download |
|:---|:---:|---:|:---|:---:|
| **NTU60-C** | X-View | 18,932 | 17 individual + auxiliary Mixed | Upon acceptance |
| **NTU120-C** | X-Sub | 50,919 | 17 individual | Upon acceptance |

> 📥 **Dataset downloads:** Public download links for NTU60-C and NTU120-C will be released upon paper acceptance. The [generation scripts](scripts/prepare_ntu120c.py) and [file manifests](data/manifests/) are available in this anonymous repository.

Main results average the 17 individual corruptions. NTU60-C recurring streams include Mixed and retain adaptation state throughout the stream.

## 🛠️ Installation

Linux with a CUDA GPU is required. ProtoGCN and PYSKL are included in this repository.

```bash
conda create -n sar-tta-c python=3.8 -y
conda activate sar-tta-c
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -r requirements.txt
```

## 🚀 Quick Start

Prepare the [data and checkpoints](#-data--checkpoints), then run from the repository root:

```bash
# NTU60-C and NTU120-C main comparisons
bash run.sh main60
bash run.sh main120

# Long recurring stream
bash run.sh long
```

<details>
<summary><b>🔬 Additional experiments</b></summary>

| Experiment | Command |
|:---|:---|
| Component ablation | `bash run.sh ablation` |
| Long-stream ablation | `bash run.sh ablation-long` |
| Auxiliary Mixed scenario | `bash run.sh mixed` |
| Unseen sensor-mounting faults | `bash run.sh heldout` |
| NTU to PKU-MMD transfer | `bash run.sh real` |
| Backbone comparisons | `bash run.sh backbone` |
| Computational cost and FLOPs | `bash run.sh cost` |

Run `bash run.sh --help` for all available experiments and options. PKU-MMD preparation is described in the [data guide](data/README.md#pku-mmd-real-shift).

</details>

Results are saved under `outputs/`. Summarize completed experiments with:

```bash
python scripts/summarize.py outputs
```

## 📦 Data & Checkpoints

See the [data guide](data/README.md) to generate the benchmarks from official annotations. Public download links for prepared datasets and checkpoints will be provided upon paper acceptance. When using prepared archives, place them and their checksum files in `release_assets/`.

<details>
<summary><b>🔐 Verify and extract the downloads</b></summary>

Run these commands from the repository root:

```bash
(cd release_assets && sha256sum -c ntu60_corruptions_v2_xview.tar.gz.SHA256SUMS)
cat release_assets/ntu60_corruptions_v2_xview.tar.gz.part* | tar -xz

(cd release_assets && sha256sum -c ntu120_corruptions_v2_xsub.tar.gz.SHA256SUMS)
cat release_assets/ntu120_corruptions_v2_xsub.tar.gz.part* | tar -xz

(cd release_assets && sha256sum -c checkpoints.tar.gz.SHA256SUMS)
tar -xzf release_assets/checkpoints.tar.gz
```

The archives extract directly into `data/nturgbd/` and `checkpoints/`.

</details>

| Resource | Contents |
|:---|:---|
| 📖 [Data guide](data/README.md) | Formats, generation, official sources, and PKU-MMD preparation |
| 🧾 [Dataset manifests](data/manifests/) | File sizes and SHA256 checksums |
| 🧠 [Checkpoint manifest](checkpoints_manifest.json) | Evaluated weights and provenance |

Prepared datasets support the main comparisons directly. Regeneration and clean-input held-out experiments additionally require the original clean annotations.

## 🤝 Acknowledgments & License

This project builds on ProtoGCN, PYSKL, and the comparison methods listed in [THIRD_PARTY.md](THIRD_PARTY.md). Original third-party notices are retained.

Release-specific code is covered by [LICENSE](LICENSE). Datasets and pretrained weights retain their original usage terms.
