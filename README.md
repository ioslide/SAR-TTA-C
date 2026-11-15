# SAR-TTA-C

Anonymous code and benchmark release for **When the Sensor Stays Broken: Test-Time Robust Skeleton-Based Action Recognition**.

The proposed method is **SRT**, implemented in `methods/SRT_TTA_paper.py`. This release includes the evaluation runners, paper baselines, component ablations, NTU60-C/NTU120-C corruption generators, and portable dataset manifests.

## Installation

The evaluated environment uses Linux, Python 3.8, PyTorch 2.0.1, CUDA 11.8, and MMCV-full 1.7.2. A CUDA GPU is required for evaluation.

```bash
conda create -n sar-tta-c python=3.8 -y
conda activate sar-tta-c
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install -r requirements.txt
```

Run commands from the repository root. ProtoGCN is included in `protogcn/`; PYSKL is included in `third_party/pyskl/`. No sibling checkout is needed.

## Data and checkpoints

Download the dataset parts and `checkpoints.tar.gz` from the accompanying release assets. Extract them in the repository root; the archives contain the required `data/nturgbd/` and `checkpoints/` paths. The code repository contains metadata and generators; the full arrays are distributed separately.

```bash
# Run checksum verification where the downloaded assets are stored.
sha256sum -c ntu60_corruptions_v2_xview.tar.gz.SHA256SUMS
cat ntu60_corruptions_v2_xview.tar.gz.part* | tar -xz
sha256sum -c ntu120_corruptions_v2_xsub.tar.gz.SHA256SUMS
cat ntu120_corruptions_v2_xsub.tar.gz.part* | tar -xz
tar -xzf checkpoints.tar.gz
```

See [the data card](data/README.md) for formats, generation, official data sources, and usage terms; see [checkpoint provenance](checkpoints_manifest.json) for the evaluated weights. Prepared corruption arrays suffice for the main evaluations; the clean source annotations are needed for regeneration and clean-input held-out transformations.

## Reproduce experiments

```bash
# Display the planned commands without running an experiment.
bash run.sh main60 --plan

# Main tables: all 13 methods, seed 42.
bash run.sh main60
bash run.sh main120

# SRT over the three reported evaluation seeds.
bash run.sh main60 --methods SRT_TTA_paper --seeds 42 43 44
bash run.sh main120 --methods SRT_TTA_paper --seeds 42 43 44

# Long recurring stream, keeping adapter state between domains and repetitions.
bash run.sh long

# Additional experiments; each command selects one group.
bash run.sh ablation
bash run.sh orientation
bash run.sh sensitivity
bash run.sh intervention
bash run.sh heldout
bash run.sh real
bash run.sh backbone
bash run.sh theory
bash run.sh vote
bash run.sh dwell
bash run.sh plugin
bash run.sh cost
```

`--methods`, `--seeds`, and `--output` select methods, evaluation seeds, and output location. `--dry-run` validates runner configurations without loading data or models. The launcher continues to the next experiment after a task fails and reports the failures at the end. Each task uses its own output directory. Runs do not edit any manuscript.

| Group | Evaluation |
|---|---|
| `main60` | NTU60-C, X-View, one pass through the fixed 18-domain sequence; primary mean excludes Mixed |
| `main120` | NTU120-C, X-Sub, one pass through 17 individual domains |
| `long` | NTU60-C recurring stream, 159,840 mini-batch time steps by default |
| `ablation`, `orientation`, `sensitivity`, `intervention` | Component, orientation, parameter, and intervention studies |
| `heldout` | Clean-input roll90, roll180, and roll180 with time reversal |
| `real` | NTU60 to PKU-MMD, 50 shared action classes, full 60-way classifier |
| `backbone` | ST-GCN++, CTR-GCN, and ProtoGCN X-Sub recurring evaluations |
| `theory`, `vote`, `dwell`, `plugin` | Gradient diagnostics, frozen-source votes, persistence scans, and ATC transfer |
| `cost` | One complete segment-dropout domain per method, including measured FLOPs |

## Evaluation protocol

Batch size is 32; each skeleton is sampled to 100 frames with one test clip. Corruptions use profile v2, severity 5, and generation seed 42. Evaluation seeds control the evaluation pipeline independently of dataset generation. Domains follow `core/runtime.py`; adapters retain their state across domain boundaries. Target labels are used by the evaluator, not supplied to adaptation methods.

NTU60-C contains 18,932 clips per domain (592 mini-batches); NTU120-C contains 50,919 (1,592 mini-batches). The primary tables average the 17 individual domains. Mixed is an auxiliary NTU60-C domain and remains in its recurring adaptation stream. One complete 18-domain NTU60-C traversal has 10,656 time steps. The compatibility option `--rounds 15` specifies 15 such traversals, or 159,840 time steps. Source is evaluated once because its state remains fixed. Learned and cache-based adapters keep state throughout the recurring stream.

| SRT setting | NTU60-C | NTU120-C |
|---|---:|---:|
| Learning rate | 0.00025 | 0.00025 |
| SGD momentum | 0.9 | 0.9 |
| Affinity power alpha | 1.75 | 1.25 |
| Weight scale beta | 3 | 1 |
| Prediction-state momentum rho_pi | 0.9 | 0.99 |
| Orientation-state momentum rho_s | 0.9 | 0.9 |
| Run note | `full` | `ntu120` |

The launcher selects these settings explicitly. `SRT` is an alias for `SRT_TTA_paper`; `Skeleton-Cache` is an alias for `SkeletonCacheUniform`. The Skeleton-Cache baseline is a causal closed-set port with uniform descriptor fusion on the same classifier. Its parameters remain frozen, while its feature cache changes with the stream.

Measurements and traces are written beneath `outputs/`. Run `python scripts/summarize.py outputs` to print completed 17-domain means and sample standard deviations across seeds; Mixed is listed separately. Raw runner CSV files retain the full evaluated domain sequence. SRT retains the compatibility instrumentation variables `ASTRA_TIME_DIM` (default 2) and `ASTRA_TRACE` (default 1). Keep the defaults for reproduction. `EVAL_CACHE_DIR` selects the disk cache for deterministic evaluation inputs; cached inputs are memory mapped. `SKIP_PKL=1` avoids redundant prediction pickle files while preserving stream measurements.

The cost group records elapsed GPU time, peak allocated memory, and convolution/matrix-multiplication FLOPs for actual forward and backward operations. One multiply-add counts as two FLOPs. This is an operator-scoped measurement, not a count of every elementary operation. Measurements depend on hardware and software versions.

## Attribution

Third-party implementations and pretrained models retain their original attribution and licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). Dataset usage remains governed by the original dataset terms. Release-specific code uses the root license.
