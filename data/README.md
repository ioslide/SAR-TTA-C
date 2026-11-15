# NTU60-C and NTU120-C

These benchmarks apply deterministic sensor-failure transformations to the official NTU RGB+D 3D skeleton annotations. They contain skeleton coordinates and action labels, without RGB videos or author-specific filesystem metadata.

| Dataset | Evaluation split | Classes | Clips per domain | Released corruption domains |
|---|---|---:|---:|---:|
| NTU60-C | X-View validation | 60 | 18,932 | 17 individual + Mixed |
| NTU120-C | X-Sub validation | 120 | 50,919 | 17 individual; clean reference also included |

The ordered individual domains are segment dropout, segment shuffle, temporal delay, temporal reversal, camera viewpoint, nonrigid joint drift, coordinate quantization, skeleton mirror, persistent joint missing, burst joint dropout, identity switch, left/right asymmetry, joint topology swap, temporally correlated noise, outlier frames, depth-axis noise, and high-frequency jitter. Mixed combines failures and is auxiliary to the NTU60-C primary mean.

## Format and integrity

Each `*.pkl.mmap/` directory contains `index.json` and `keypoints.dat`. The format is `protogcn_annotation_mmap`, version 1. Coordinates are float16; the index stores shapes, offsets, labels, clip identifiers, and the evaluation split. Shapes retain variable-length sequences and the original person/joint organization. Sampling and normalization take place during evaluation.

The catalogs in `data/manifests/` list the released files, sizes, and SHA256 digests. Archive parts are at most 1 GiB and have separate SHA256 lists. Concatenate all numerically ordered parts before extraction. All coordinates and sample orders match the prepared evaluation data; only source paths and archive ownership/timestamps are made portable. NTU120 Mixed is excluded from this release.

## Generate from official annotations

Obtain the original NTU skeleton annotations through the [NTU RGB+D project](https://rose1.ntu.edu.sg/dataset/actionRecognition/) and the [ProtoGCN data preparation instructions](https://github.com/firework8/ProtoGCN). Place `ntu60_3danno.pkl` and `ntu120_3danno.pkl` in `data/nturgbd/`, or edit the relative paths in the dataset YAML files.

```bash
python scripts/prepare_ntu120c.py --dataset-config configs/dataset/ntu60_xview.yaml
python scripts/prepare_ntu120c.py --dataset-config configs/dataset/ntu120_xsub.yaml
```

Despite its historical filename, this script supports both class counts. It filters to the configured evaluation split before generating test-only memory-mapped domains. Randomness is keyed by generation seed, corruption name, and clip identifier. The scripts preserve the existing transform computations and float16 storage. Original full annotations can require substantial host memory during generation; evaluation reads the prepared arrays on demand.

For the optional ProtoGCN X-Sub backbone experiment, generate the NTU60 X-Sub domains separately:

```bash
python scripts/prepare_ntu120c.py --dataset-config configs/dataset/ntu60_xsub_skeleton_c_v2.yaml
```

The general-purpose `generate_offline_corruptions.py` can also generate pickle annotations. Use `--profile v2` to reproduce the benchmark version; the split-filtering mmap generator above is preferred for evaluation.

## PKU-MMD real shift

Download Part I skeletons and annotations from the [official project](http://39.96.165.147/Projects/PKUMMD/PKU-MMD.html). Put `Skeleton.7z`, `Label_PKUMMD.7z`, and `Actions.xlsx` under `data/real_shifts/pku/sources/`. Install the optional preparation dependencies and libarchive on the host, then run:

```bash
pip install -r requirements-pku.txt
python scripts/prepare_pku_real.py
bash run.sh real
```

Preparation retains 20,621 valid action segments from the 50 shared classes. It preserves camera-space XYZ, NTU25 joint order, both person slots, and float32 coordinates. The output includes an explicit class mapping, exclusions, source checksums, and a portable dataset configuration. The stream follows video-name order and original annotation-row order. Raw PKU archives are obtained separately under their original terms.

## Terms

The NTU-derived coordinates and PKU data remain subject to the original datasets' access and usage terms. The software license does not relicense those datasets or their pretrained weights. Consult the original providers before redistribution or use beyond the permitted research setting.
