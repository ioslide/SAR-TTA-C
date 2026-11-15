#!/usr/bin/env python3
"""Generate reproducible offline NTU RGB+D skeleton corruption datasets.

The output files preserve the original ``split`` and annotation schema used by
``PoseDataset``.  Only each annotation's ``keypoint`` array is replaced, so an
output file can be supplied directly as ``dataset.ann_file``.
"""

import argparse
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path('data/nturgbd/ntu60_3danno.pkl')
V1_OUTPUT_DIR = Path('data/nturgbd/ntu60_corruptions_v1')
DEFAULT_OUTPUT_DIR = Path('data/nturgbd/ntu60_corruptions_v2')
V2_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
DEFAULT_SEVERITY = 5
DEFAULT_SEED = 42
FORMAT_VERSION = 'ntu60-offline-corruptions-v1'
V2_FORMAT_VERSION = 'skeleton-c-v2'

# NTU RGB+D 25-joint tree, represented as (child, parent), zero indexed.
NTU_BONES = [
    (1, 0), (20, 1), (2, 20), (3, 2),
    (4, 20), (5, 4), (6, 5), (7, 6), (21, 7), (22, 7),
    (8, 20), (9, 8), (10, 9), (11, 10), (23, 11), (24, 11),
    (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18),
]
LEFT_BONES = {4, 5, 6, 7, 21, 22, 12, 13, 14, 15}
RIGHT_BONES = {8, 9, 10, 11, 23, 24, 16, 17, 18, 19}
LEFT_RIGHT_PAIRS = [(4, 8), (5, 9), (6, 10), (7, 11), (21, 23), (22, 24),
                    (12, 16), (13, 17), (14, 18), (15, 19)]
LIMBS = [
    [4, 5, 6, 7, 21, 22],
    [8, 9, 10, 11, 23, 24],
    [12, 13, 14, 15],
    [16, 17, 18, 19],
]


def _strength(severity):
    if severity not in range(1, 6):
        raise ValueError('severity must be an integer from 1 to 5')
    return severity / 5.0


def _rng_for_item(seed, corruption, frame_dir):
    digest = hashlib.blake2b(
        f'{seed}|{corruption}|{frame_dir}'.encode('utf-8'), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, byteorder='little', signed=False))


def _valid_mask(keypoint):
    return np.any(np.abs(keypoint) > 1e-6, axis=-1)


def _restore_invalid(keypoint, valid):
    keypoint[~valid] = 0.0
    return keypoint


def _centers(keypoint, valid):
    """Return a valid per-person, per-frame reference point for relative transforms."""
    roots = keypoint[:, :, 1, :].copy()
    root_valid = valid[:, :, 1]
    counts = valid.sum(axis=2, keepdims=True).clip(min=1)
    fallback = (keypoint * valid[..., None]).sum(axis=2) / counts
    roots[~root_valid] = fallback[~root_valid]
    return roots[:, :, None, :]


def _rotation_matrix(angles):
    x, y, z = angles
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _interpolate_time(keypoint, positions):
    """Sample a sequence at floating-point positions while preserving its length."""
    frames = keypoint.shape[1]
    lower = np.floor(np.clip(positions, 0, frames - 1)).astype(np.int64)
    upper = np.minimum(lower + 1, frames - 1)
    weights = (positions - lower).astype(np.float32)
    return keypoint[:, lower] * (1.0 - weights)[None, :, None, None] + keypoint[:, upper] * weights[None, :, None, None]


def random_speed(keypoint, rng, severity):
    frames = keypoint.shape[1]
    if frames < 2:
        return keypoint.copy()
    scale = _strength(severity)
    speed = rng.uniform(1.0 - 0.35 * scale, 1.0 + 0.35 * scale)
    center = (frames - 1) / 2.0
    return _interpolate_time(keypoint, center + (np.arange(frames) - center) * speed)


def local_time_warp(keypoint, rng, severity):
    frames = keypoint.shape[1]
    if frames < 4:
        return keypoint.copy()
    scale = _strength(severity)
    positions = np.arange(frames, dtype=np.float32)
    start = int(rng.integers(0, max(1, frames // 3)))
    width = int(max(2, round(frames * rng.uniform(0.2, 0.35 + 0.15 * scale))))
    end = min(frames, start + width)
    local = np.linspace(0.0, np.pi, end - start, dtype=np.float32)
    shift = rng.uniform(-1.0, 1.0) * (1.0 + 5.0 * scale) * np.sin(local)
    positions[start:end] += shift
    return _interpolate_time(keypoint, positions)


def segment_dropout(keypoint, rng, severity):
    out = keypoint.copy()
    frames = out.shape[1]
    length = min(frames, max(1, int(round(frames * (0.04 + 0.31 * _strength(severity))))))
    start = int(rng.integers(0, frames - length + 1))
    out[:, start:start + length] = 0.0
    return out


def segment_freeze(keypoint, rng, severity):
    out = keypoint.copy()
    frames = out.shape[1]
    length = min(frames, max(1, int(round(frames * (0.05 + 0.30 * _strength(severity))))))
    start = int(rng.integers(0, frames - length + 1))
    source = out[:, start:start + 1].copy()
    out[:, start:start + length] = source
    return out


def segment_shuffle(keypoint, rng, severity):
    frames = keypoint.shape[1]
    segments = min(frames, 2 + int(round(4 * _strength(severity))))
    if segments < 2:
        return keypoint.copy()
    boundaries = np.linspace(0, frames, segments + 1, dtype=np.int64)
    chunks = [keypoint[:, boundaries[i]:boundaries[i + 1]] for i in range(segments)]
    return np.concatenate([chunks[index] for index in rng.permutation(segments)], axis=1)


def temporal_delay(keypoint, rng, severity):
    out = keypoint.copy()
    frames = out.shape[1]
    delay = min(frames - 1, max(1, int(round(frames * (0.02 + 0.18 * _strength(severity))))))
    if delay <= 0:
        return out
    out[:, delay:] = keypoint[:, :-delay]
    out[:, :delay] = keypoint[:, :1]
    return out


def temporal_reverse(keypoint, rng, severity):
    del rng, severity
    return keypoint[:, ::-1].copy()


def spatial_translation(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    scale = _strength(severity)
    trajectory = np.linspace(0.0, 1.0, out.shape[1], dtype=np.float32)[None, :, None, None]
    offset = rng.uniform(-0.05 - 0.20 * scale, 0.05 + 0.20 * scale, size=(out.shape[0], 1, 1, 3)).astype(np.float32)
    out += trajectory * offset * valid[..., None]
    return _restore_invalid(out, valid)


def spatial_scaling(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    factor = rng.uniform(1.0 - 0.12 * _strength(severity), 1.0 + 0.12 * _strength(severity))
    center = _centers(out, valid)
    out = center + (out - center) * factor
    return _restore_invalid(out, valid)


def nonrigid_joint_drift(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    scale = _strength(severity)
    frames = out.shape[1]
    phases = rng.uniform(0, 2 * np.pi, size=(out.shape[0], 1, out.shape[2], 3)).astype(np.float32)
    amplitudes = rng.normal(0.0, 0.015 + 0.075 * scale, size=(out.shape[0], 1, out.shape[2], 3)).astype(np.float32)
    time = np.linspace(0, 2 * np.pi, frames, dtype=np.float32)[None, :, None, None]
    drift = amplitudes * np.sin(time + phases)
    drift[:, :, 1, :] = 0.0
    out += drift * valid[..., None]
    return _restore_invalid(out, valid)


def coordinate_quantization(keypoint, rng, severity):
    del rng
    step = 0.005 + 0.045 * _strength(severity)
    valid = _valid_mask(keypoint)
    out = np.round(keypoint / step) * step
    return _restore_invalid(out, valid)


def coordinate_clipping(keypoint, rng, severity):
    del rng
    out = keypoint.copy()
    valid = _valid_mask(out)
    center = _centers(out, valid)
    radius = 1.10 - 0.55 * _strength(severity)
    out = center + np.clip(out - center, -radius, radius)
    return _restore_invalid(out, valid)


def camera_viewpoint(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    angle = 0.08 + 0.52 * _strength(severity)
    matrix = _rotation_matrix(rng.uniform(-angle, angle, size=3))
    center = _centers(out, valid)
    out = center + np.einsum('ij,mtvj->mtvi', matrix, out - center)
    return _restore_invalid(out, valid)


def perspective_distortion(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    center = _centers(out, valid)
    relative = out - center
    strength = rng.uniform(0.08, 0.35) * _strength(severity)
    denom = np.clip(1.0 + strength * relative[..., 2:3], 0.45, 1.8)
    relative[..., :2] /= denom
    out = center + relative
    return _restore_invalid(out, valid)


def person_dropout(keypoint, rng, severity):
    out = keypoint.copy()
    people = out.shape[0]
    count = min(people, 1 + int(rng.random() < 0.35 * _strength(severity)))
    for person in rng.choice(people, size=count, replace=False):
        out[person] = 0.0
    return out


def limb_occlusion(keypoint, rng, severity):
    out = keypoint.copy()
    count = min(len(LIMBS), 1 + int(round(2 * _strength(severity))))
    for limb_index in rng.choice(len(LIMBS), size=count, replace=False):
        out[:, :, LIMBS[limb_index], :] = 0.0
    return out


def persistent_joint_missing(keypoint, rng, severity):
    out = keypoint.copy()
    candidates = np.array([index for index in range(out.shape[2]) if index not in {0, 1, 20}])
    count = min(len(candidates), 1 + int(round(7 * _strength(severity))))
    joints = rng.choice(candidates, size=count, replace=False)
    out[:, :, joints, :] = 0.0
    return out


def local_joint_masking(keypoint, rng, severity):
    out = keypoint.copy()
    frames = out.shape[1]
    length = min(frames, max(1, int(round(frames * (0.06 + 0.32 * _strength(severity))))))
    start = int(rng.integers(0, frames - length + 1))
    joint_count = min(out.shape[2], 2 + int(round(8 * _strength(severity))))
    joints = rng.choice(out.shape[2], size=joint_count, replace=False)
    out[:, start:start + length, joints, :] = 0.0
    return out


def _scale_bones(keypoint, bone_scales):
    out = keypoint.copy()
    valid = _valid_mask(keypoint)
    for child, parent in NTU_BONES:
        if child >= out.shape[2] or parent >= out.shape[2]:
            continue
        pair_valid = valid[:, :, child] & valid[:, :, parent]
        vector = keypoint[:, :, child] - keypoint[:, :, parent]
        out[:, :, child] = np.where(
            pair_valid[..., None],
            out[:, :, parent] + vector * bone_scales[child],
            out[:, :, child],
        )
    return _restore_invalid(out, valid)


def bone_length_scaling(keypoint, rng, severity):
    scales = np.ones(keypoint.shape[2], dtype=np.float32)
    offset = 0.08 + 0.20 * _strength(severity)
    scales += rng.uniform(-offset, offset, size=scales.shape).astype(np.float32)
    scales[0] = 1.0
    return _scale_bones(keypoint, scales)


def left_right_asymmetry(keypoint, rng, severity):
    scales = np.ones(keypoint.shape[2], dtype=np.float32)
    delta = rng.uniform(0.06, 0.28 * _strength(severity) + 0.06)
    for joint in LEFT_BONES:
        if joint < len(scales):
            scales[joint] = 1.0 + delta
    for joint in RIGHT_BONES:
        if joint < len(scales):
            scales[joint] = 1.0 - delta
    return _scale_bones(keypoint, scales)


def joint_topology_swap(keypoint, rng, severity):
    out = keypoint.copy()
    count = min(len(LEFT_RIGHT_PAIRS), 1 + int(round(7 * _strength(severity))))
    for pair_index in rng.choice(len(LEFT_RIGHT_PAIRS), size=count, replace=False):
        left, right = LEFT_RIGHT_PAIRS[pair_index]
        if right < out.shape[2]:
            out[:, :, [left, right], :] = out[:, :, [right, left], :]
    return out


def skeleton_center_drift(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    frames = out.shape[1]
    scale = _strength(severity)
    endpoint = rng.uniform(-0.03 - 0.20 * scale, 0.03 + 0.20 * scale, size=(out.shape[0], 1, 1, 3)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=(out.shape[0], 1, 1, 3)).astype(np.float32)
    time = np.linspace(0, 1, frames, dtype=np.float32)[None, :, None, None]
    drift = endpoint * time + 0.025 * scale * np.sin(2 * np.pi * time + phase)
    out += drift * valid[..., None]
    return _restore_invalid(out, valid)


def temporal_correlated_noise(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    noise = rng.normal(0.0, 1.0, size=out.shape).astype(np.float32)
    noise = np.cumsum(noise, axis=1)
    noise /= np.maximum(np.max(np.abs(noise), axis=1, keepdims=True), 1e-6)
    out += noise * (0.015 + 0.10 * _strength(severity)) * valid[..., None]
    return _restore_invalid(out, valid)


def low_frequency_drift(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    time = np.linspace(0, 1, out.shape[1], dtype=np.float32)[None, :, None, None]
    freq = rng.uniform(0.35, 1.5, size=(out.shape[0], 1, 1, 3)).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, size=(out.shape[0], 1, 1, 3)).astype(np.float32)
    amplitude = rng.uniform(0.01, 0.11 * _strength(severity) + 0.01, size=(out.shape[0], 1, 1, 3)).astype(np.float32)
    out += amplitude * np.sin(2 * np.pi * freq * time + phase) * valid[..., None]
    return _restore_invalid(out, valid)


def sensor_jitter(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    jitter = rng.normal(0.0, 0.006 + 0.055 * _strength(severity), size=out.shape).astype(np.float32)
    out += jitter * valid[..., None]
    return _restore_invalid(out, valid)


def outlier_frames(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    frames = out.shape[1]
    count = min(frames, 1 + int(round(frames * 0.015 * severity)))
    selected_frames = rng.choice(frames, size=count, replace=False)
    joint_count = min(out.shape[2], 1 + int(round(3 * _strength(severity))))
    joints = rng.choice(out.shape[2], size=joint_count, replace=False)
    spikes = rng.normal(0.0, 0.16 + 0.42 * _strength(severity), size=(out.shape[0], count, joint_count, 3)).astype(np.float32)
    selection_valid = valid[:, selected_frames[:, None], joints[None, :]]
    out[:, selected_frames[:, None], joints[None, :], :] += spikes * selection_valid[..., None]
    return _restore_invalid(out, valid)


def depth_axis_noise(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    if out.shape[-1] >= 3:
        out[..., 2] += rng.normal(0.0, 0.015 + 0.13 * _strength(severity), size=out.shape[:-1]).astype(np.float32) * valid
    return _restore_invalid(out, valid)


# Skeleton-C v2 uses explicit, higher-impact failure modes.  The severity-5
# values are calibrated to represent a severe but still plausible sensor or
# tracking failure; v1 remains unchanged for backwards compatibility.
def v2_random_speed(keypoint, rng, severity):
    frames = keypoint.shape[1]
    if frames < 2:
        return keypoint.copy()
    speed = rng.uniform(0.55, 1.55) if severity == 5 else rng.uniform(
        1.0 - 0.10 * severity, 1.0 + 0.10 * severity)
    center = (frames - 1) / 2.0
    return _interpolate_time(keypoint, center + (np.arange(frames) - center) * speed)


def v2_local_time_warp(keypoint, rng, severity):
    frames = keypoint.shape[1]
    if frames < 4:
        return keypoint.copy()
    scale = _strength(severity)
    positions = np.arange(frames, dtype=np.float32)
    start = int(rng.integers(0, max(1, frames // 3)))
    width = int(max(2, round(frames * (0.25 + 0.35 * scale))))
    end = min(frames, start + width)
    local = np.linspace(0.0, np.pi, end - start, dtype=np.float32)
    shift = rng.uniform(-1.0, 1.0) * (3.0 + 10.0 * scale) * np.sin(local)
    positions[start:end] += shift
    return _interpolate_time(keypoint, positions)


def v2_segment_dropout(keypoint, rng, severity):
    out = keypoint.copy()
    frames = out.shape[1]
    fraction = 0.10 + 0.45 * _strength(severity)
    length = min(frames, max(1, int(round(frames * fraction))))
    start = int(rng.integers(0, frames - length + 1))
    out[:, start:start + length] = 0.0
    return out


def v2_segment_shuffle(keypoint, rng, severity):
    segments = min(keypoint.shape[1], 3 + int(round(5 * _strength(severity))))
    boundaries = np.linspace(0, keypoint.shape[1], segments + 1, dtype=np.int64)
    chunks = [keypoint[:, boundaries[i]:boundaries[i + 1]] for i in range(segments)]
    return np.concatenate([chunks[index] for index in rng.permutation(segments)], axis=1)


def v2_temporal_delay(keypoint, rng, severity):
    del rng
    out = keypoint.copy()
    frames = out.shape[1]
    delay = min(frames - 1, max(1, int(round(frames * (0.08 + 0.32 * _strength(severity))))))
    out[:, delay:] = keypoint[:, :-delay]
    out[:, :delay] = keypoint[:, :1]
    return out


def v2_camera_viewpoint(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    angle = 0.30 + 0.90 * _strength(severity)
    matrix = _rotation_matrix(rng.uniform(-angle, angle, size=3))
    center = _centers(out, valid)
    out = center + np.einsum('ij,mtvj->mtvi', matrix, out - center)
    return _restore_invalid(out, valid)


def v2_perspective_distortion(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    center = _centers(out, valid)
    relative = out - center
    strength = rng.uniform(0.25, 0.85) * _strength(severity)
    denom = np.clip(1.0 + strength * relative[..., 2:3], 0.25, 2.5)
    relative[..., :2] /= denom
    return _restore_invalid(center + relative, valid)


def v2_nonrigid_joint_drift(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    frames = out.shape[1]
    scale = _strength(severity)
    phases = rng.uniform(0, 2 * np.pi, size=(out.shape[0], 1, out.shape[2], 3)).astype(np.float32)
    amplitudes = rng.normal(0.0, 0.04 + 0.18 * scale, size=(out.shape[0], 1, out.shape[2], 3)).astype(np.float32)
    time = np.linspace(0, 2 * np.pi, frames, dtype=np.float32)[None, :, None, None]
    drift = amplitudes * np.sin(time + phases)
    drift[:, :, 1, :] = 0.0
    out += drift * valid[..., None]
    return _restore_invalid(out, valid)


def v2_coordinate_quantization(keypoint, rng, severity):
    del rng
    step = 0.02 + 0.10 * _strength(severity)
    valid = _valid_mask(keypoint)
    return _restore_invalid(np.round(keypoint / step) * step, valid)


def v2_limb_occlusion(keypoint, rng, severity):
    out = keypoint.copy()
    count = min(len(LIMBS), 1 + int(round(3 * _strength(severity))))
    for limb_index in rng.choice(len(LIMBS), size=count, replace=False):
        out[:, :, LIMBS[limb_index], :] = 0.0
    return out


def v2_persistent_joint_missing(keypoint, rng, severity):
    out = keypoint.copy()
    candidates = np.array([index for index in range(out.shape[2]) if index not in {0, 1, 20}])
    count = min(len(candidates), 2 + int(round(10 * _strength(severity))))
    out[:, :, rng.choice(candidates, size=count, replace=False), :] = 0.0
    return out


def v2_person_dropout(keypoint, rng, severity):
    out = keypoint.copy()
    if out.shape[0] > 1 and severity >= 4:
        out[int(rng.integers(0, out.shape[0]))] = 0.0
    return out


def v2_bone_length_scaling(keypoint, rng, severity):
    scales = np.ones(keypoint.shape[2], dtype=np.float32)
    offset = 0.18 + 0.42 * _strength(severity)
    scales += rng.uniform(-offset, offset, size=scales.shape).astype(np.float32)
    scales[0] = 1.0
    return _scale_bones(keypoint, scales)


def v2_left_right_asymmetry(keypoint, rng, severity):
    scales = np.ones(keypoint.shape[2], dtype=np.float32)
    delta = rng.uniform(0.18, 0.48 * _strength(severity) + 0.18)
    for joint in LEFT_BONES:
        if joint < len(scales):
            scales[joint] = 1.0 + delta
    for joint in RIGHT_BONES:
        if joint < len(scales):
            scales[joint] = 1.0 - delta
    return _scale_bones(keypoint, scales)


def v2_joint_topology_swap(keypoint, rng, severity):
    out = keypoint.copy()
    count = min(len(LEFT_RIGHT_PAIRS), 2 + int(round(8 * _strength(severity))))
    for pair_index in rng.choice(len(LEFT_RIGHT_PAIRS), size=count, replace=False):
        left, right = LEFT_RIGHT_PAIRS[pair_index]
        out[:, :, [left, right], :] = out[:, :, [right, left], :]
    return out


def v2_temporal_correlated_noise(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    noise = np.cumsum(rng.normal(0.0, 1.0, size=out.shape).astype(np.float32), axis=1)
    noise /= np.maximum(np.max(np.abs(noise), axis=1, keepdims=True), 1e-6)
    out += noise * (0.04 + 0.20 * _strength(severity)) * valid[..., None]
    return _restore_invalid(out, valid)


def v2_outlier_frames(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    frames = out.shape[1]
    count = min(frames, max(1, int(round(frames * (0.03 + 0.07 * _strength(severity))))))
    selected_frames = rng.choice(frames, size=count, replace=False)
    joint_count = min(out.shape[2], 3 + int(round(8 * _strength(severity))))
    joints = rng.choice(out.shape[2], size=joint_count, replace=False)
    spikes = rng.normal(0.0, 0.35 + 0.85 * _strength(severity), size=(out.shape[0], count, joint_count, 3)).astype(np.float32)
    selection_valid = valid[:, selected_frames[:, None], joints[None, :]]
    out[:, selected_frames[:, None], joints[None, :], :] += spikes * selection_valid[..., None]
    return _restore_invalid(out, valid)


def v2_depth_axis_noise(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    if out.shape[-1] >= 3:
        out[..., 2] += rng.normal(0.0, 0.04 + 0.24 * _strength(severity), size=out.shape[:-1]).astype(np.float32) * valid
    return _restore_invalid(out, valid)


def v2_temporal_reverse(keypoint, rng, severity):
    del rng, severity
    return keypoint[:, ::-1].copy()


def v2_frame_blur(keypoint, rng, severity):
    """Temporal averaging simulating low-FPS motion blur and interpolation."""
    del rng
    out = keypoint.copy()
    if out.shape[1] < 3:
        return out
    previous = np.concatenate([out[:, :1], out[:, :-1]], axis=1)
    following = np.concatenate([out[:, 1:], out[:, -1:]], axis=1)
    strength = _strength(severity)
    return (0.5 * out + 0.25 * strength * previous + 0.25 * strength * following) / (0.5 + 0.5 * strength)


def v2_joint_dropout_burst(keypoint, rng, severity):
    """A tracker loses many joints for one long contiguous time interval."""
    out = keypoint.copy()
    frames = out.shape[1]
    length = min(frames, max(1, int(round(frames * (0.25 + 0.35 * _strength(severity))))))
    start = int(rng.integers(0, frames - length + 1))
    candidates = np.array([index for index in range(out.shape[2]) if index not in {0, 1, 20}])
    count = min(len(candidates), 5 + int(round(9 * _strength(severity))))
    joints = rng.choice(candidates, size=count, replace=False)
    out[:, start:start + length, joints, :] = 0.0
    return out


def v2_identity_switch(keypoint, rng, severity):
    """Swap tracked-person identities after a random transition point."""
    del severity
    out = keypoint.copy()
    if out.shape[0] < 2:
        return out
    transition = int(rng.integers(max(1, out.shape[1] // 4), max(2, 3 * out.shape[1] // 4)))
    first, second = int(rng.integers(0, out.shape[0])), int(rng.integers(0, out.shape[0]))
    if first == second:
        second = (first + 1) % out.shape[0]
    out[[first, second], transition:] = out[[second, first], transition:]
    return out


def v2_skeleton_mirror(keypoint, rng, severity):
    """Mirror the skeleton around its root and exchange left/right joints."""
    del rng, severity
    out = keypoint.copy()
    valid = _valid_mask(out)
    center = _centers(out, valid)
    out[..., 0] = 2.0 * center[..., 0] - out[..., 0]
    for left, right in LEFT_RIGHT_PAIRS:
        out[:, :, [left, right], :] = out[:, :, [right, left], :]
    return _restore_invalid(out, valid)


def v2_high_frequency_jitter(keypoint, rng, severity):
    out = keypoint.copy()
    valid = _valid_mask(out)
    jitter = rng.normal(0.0, 0.08 + 0.25 * _strength(severity), size=out.shape).astype(np.float32)
    out += jitter * valid[..., None]
    return _restore_invalid(out, valid)


def v2_mixed_failure(keypoint, rng, severity):
    """Compound camera, sensor, and tracker failure for a realistic hard case."""
    out = v2_camera_viewpoint(keypoint, rng, severity)
    out = v2_temporal_correlated_noise(out, rng, severity)
    return v2_joint_dropout_burst(out, rng, severity)


CORRUPTIONS = {
    'random_speed': ('temporal', 'Random global action-speed resampling.', random_speed),
    'local_time_warp': ('temporal', 'Local nonlinear temporal displacement.', local_time_warp),
    'segment_dropout': ('temporal', 'Contiguous sequence segment replaced by missing frames.', segment_dropout),
    'segment_freeze': ('temporal', 'Contiguous sequence segment frozen to one pose.', segment_freeze),
    'segment_shuffle': ('temporal', 'Temporal chunks permuted within a sequence.', segment_shuffle),
    'temporal_delay': ('temporal', 'Delayed skeleton stream with initial-frame padding.', temporal_delay),
    'temporal_reverse': ('temporal', 'Complete temporal reversal.', temporal_reverse),
    'spatial_translation': ('spatial', 'Time-varying global coordinate translation.', spatial_translation),
    'spatial_scaling': ('spatial', 'Root-relative global skeleton scaling.', spatial_scaling),
    'nonrigid_joint_drift': ('spatial', 'Joint-specific smooth spatial drift.', nonrigid_joint_drift),
    'coordinate_quantization': ('spatial', 'Coordinate resolution reduction.', coordinate_quantization),
    'coordinate_clipping': ('spatial', 'Root-relative coordinate saturation.', coordinate_clipping),
    'camera_viewpoint': ('spatial', 'Camera rotation around a root-relative skeleton.', camera_viewpoint),
    'perspective_distortion': ('spatial', 'Depth-dependent perspective coordinate warp.', perspective_distortion),
    'person_dropout': ('occlusion', 'One or more tracked people removed.', person_dropout),
    'limb_occlusion': ('occlusion', 'Whole limbs removed for all frames.', limb_occlusion),
    'persistent_joint_missing': ('occlusion', 'Selected joints removed for the full sequence.', persistent_joint_missing),
    'local_joint_masking': ('occlusion', 'Selected joints removed during one time window.', local_joint_masking),
    'bone_length_scaling': ('structural', 'Per-bone length scaling through the NTU kinematic tree.', bone_length_scaling),
    'left_right_asymmetry': ('structural', 'Opposite scale changes for left and right limbs.', left_right_asymmetry),
    'joint_topology_swap': ('structural', 'Left/right joint locations exchanged.', joint_topology_swap),
    'skeleton_center_drift': ('structural', 'Smooth sequence-level skeleton-center drift.', skeleton_center_drift),
    'temporal_correlated_noise': ('sensor', 'Temporally correlated coordinate noise.', temporal_correlated_noise),
    'low_frequency_drift': ('sensor', 'Low-frequency sensor baseline drift.', low_frequency_drift),
    'sensor_jitter': ('sensor', 'High-frequency coordinate jitter.', sensor_jitter),
    'outlier_frames': ('sensor', 'Sparse frame and joint outliers.', outlier_frames),
    'depth_axis_noise': ('sensor', 'Noise applied only to the depth coordinate.', depth_axis_noise),
}

V2_CORRUPTIONS = {
    'random_speed': ('temporal', 'Severe global action-speed resampling.', v2_random_speed),
    'local_time_warp': ('temporal', 'Severe local nonlinear temporal displacement.', v2_local_time_warp),
    'segment_dropout': ('temporal', 'Severe contiguous sequence dropout.', v2_segment_dropout),
    'segment_shuffle': ('temporal', 'Severe temporal chunk permutation.', v2_segment_shuffle),
    'temporal_delay': ('temporal', 'Severe delayed skeleton stream.', v2_temporal_delay),
    'camera_viewpoint': ('spatial', 'Severe camera viewpoint rotation.', v2_camera_viewpoint),
    'perspective_distortion': ('spatial', 'Severe depth-dependent perspective warp.', v2_perspective_distortion),
    'nonrigid_joint_drift': ('spatial', 'Severe joint-specific smooth drift.', v2_nonrigid_joint_drift),
    'coordinate_quantization': ('spatial', 'Severe coordinate resolution reduction.', v2_coordinate_quantization),
    'limb_occlusion': ('occlusion', 'Three-limb full-sequence occlusion at severity 5.', v2_limb_occlusion),
    'persistent_joint_missing': ('occlusion', 'Severe persistent joint tracking loss.', v2_persistent_joint_missing),
    'person_dropout': ('occlusion', 'Tracked-person dropout.', v2_person_dropout),
    'bone_length_scaling': ('structural', 'Severe per-bone length scaling.', v2_bone_length_scaling),
    'left_right_asymmetry': ('structural', 'Severe left-right limb asymmetry.', v2_left_right_asymmetry),
    'joint_topology_swap': ('structural', 'Severe left-right topology swaps.', v2_joint_topology_swap),
    'temporal_correlated_noise': ('sensor', 'Severe temporally correlated coordinate noise.', v2_temporal_correlated_noise),
    'outlier_frames': ('sensor', 'Dense large-magnitude frame and joint outliers.', v2_outlier_frames),
    'depth_axis_noise': ('sensor', 'Severe depth-axis sensor noise.', v2_depth_axis_noise),
    'temporal_reverse': ('temporal', 'Complete temporal reversal.', v2_temporal_reverse),
    'frame_blur': ('temporal', 'Low-FPS temporal motion blur.', v2_frame_blur),
    'joint_dropout_burst': ('occlusion', 'Long contiguous multi-joint tracker dropout.', v2_joint_dropout_burst),
    'identity_switch': ('tracking', 'Multi-person identity switch after a track transition.', v2_identity_switch),
    'skeleton_mirror': ('spatial', 'Mirrored camera view with left-right joint exchange.', v2_skeleton_mirror),
    'high_frequency_jitter': ('sensor', 'Severe high-frequency pose-estimation jitter.', v2_high_frequency_jitter),
    'mixed_failure': ('compound', 'Camera, sensor, and tracker failure combined.', v2_mixed_failure),
}


def _output_name(source, corruption, severity, seed):
    return f'{source.stem}_corrupt_{corruption}_s{severity}_seed{seed}.pkl'


def _write_pickle_atomic(payload, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='wb', dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False)
    temp_path = Path(handle.name)
    try:
        with handle:
            pickle.dump(payload, handle, protocol=4)
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_source(data):
    if not isinstance(data, dict) or 'annotations' not in data or 'split' not in data:
        raise ValueError('source must be a ProtoGCN annotation PKL with annotations and split keys')
    if not data['annotations']:
        raise ValueError('source annotations are empty')
    keypoint = data['annotations'][0].get('keypoint')
    if not isinstance(keypoint, np.ndarray) or keypoint.ndim != 4:
        raise ValueError('annotations must contain keypoint arrays with shape (M, T, V, C)')


def generate_one(source, output_dir, corruption, severity, seed, overwrite=False,
                 corruption_table=None, format_version=None):
    corruption_table = corruption_table or CORRUPTIONS
    category, description, transform = corruption_table[corruption]
    format_version = format_version or FORMAT_VERSION
    target = output_dir / _output_name(source, corruption, severity, seed)
    if target.exists() and not overwrite:
        return {'corruption': corruption, 'file': target.name, 'status': 'ready_existing'}

    with source.open('rb') as handle:
        data = pickle.load(handle)
    _validate_source(data)

    total_delta = 0.0
    total_values = 0
    zero_values = 0
    for annotation in data['annotations']:
        frame_dir = annotation.get('frame_dir', '')
        original = annotation['keypoint'].astype(np.float32, copy=False)
        corrupted = transform(original, _rng_for_item(seed, corruption, frame_dir), severity)
        if corrupted.shape != original.shape:
            raise RuntimeError(f'{corruption} changed keypoint shape for {frame_dir}')
        if not np.isfinite(corrupted).all():
            raise RuntimeError(f'{corruption} produced non-finite values for {frame_dir}')
        total_delta += float(np.abs(corrupted - original).sum())
        total_values += corrupted.size
        zero_values += int(np.count_nonzero(corrupted == 0.0))
        annotation['keypoint'] = corrupted.astype(np.float16)

    data['offline_corruption'] = {
        'format_version': format_version,
        'name': corruption,
        'category': category,
        'description': description,
        'severity': severity,
        'seed': seed,
        'source': str(source),
    }
    _write_pickle_atomic(data, target)
    return {
        'corruption': corruption,
        'category': category,
        'file': target.name,
        'status': 'generated',
        'mean_abs_delta': total_delta / max(1, total_values),
        'zero_fraction': zero_values / max(1, total_values),
        'annotations': len(data['annotations']),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--profile', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--severity', type=int, default=DEFAULT_SEVERITY)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--corruptions', nargs='+', choices=sorted(set(CORRUPTIONS) | set(V2_CORRUPTIONS)))
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f'source annotation file not found: {args.source}')
    _strength(args.severity)
    if args.profile == 'v2' and args.output_dir == V1_OUTPUT_DIR:
        args.output_dir = V2_OUTPUT_DIR
    elif args.profile == 'v1' and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = V1_OUTPUT_DIR
    corruption_table = V2_CORRUPTIONS if args.profile == 'v2' else CORRUPTIONS
    format_version = V2_FORMAT_VERSION if args.profile == 'v2' else FORMAT_VERSION
    selected = args.corruptions or list(corruption_table)
    unknown = [name for name in selected if name not in corruption_table]
    if unknown:
        raise ValueError(f'corruptions not available in profile {args.profile}: {unknown}')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    manifest_path = args.output_dir / f'manifest_s{args.severity}_seed{args.seed}.json'
    previous_records = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
            if (previous.get('format_version') == format_version
                    and previous.get('profile') == args.profile
                    and previous.get('severity') == args.severity
                    and previous.get('seed') == args.seed):
                previous_records = {
                    item.get('corruption'): item
                    for item in previous.get('corruptions', [])
                    if item.get('corruption')
                }
        except (OSError, ValueError, TypeError):
            previous_records = {}
    for index, corruption in enumerate(selected, start=1):
        print(f'[{index}/{len(selected)}] {corruption}', flush=True)
        record = generate_one(
            args.source, args.output_dir, corruption, args.severity, args.seed, args.overwrite,
            corruption_table=corruption_table, format_version=format_version)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    combined_records = dict(previous_records)
    combined_records.update({item['corruption']: item for item in records})
    ordered_records = [
        combined_records[name] for name in corruption_table if name in combined_records
    ]
    manifest = {
        'format_version': format_version,
        'profile': args.profile,
        'source': str(args.source),
        'severity': args.severity,
        'seed': args.seed,
        'annotation_count': len(pickle.load(args.source.open('rb'))['annotations']),
        'corruptions': ordered_records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n')
    print(f'manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
