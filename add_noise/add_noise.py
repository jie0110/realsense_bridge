import random as _random
from typing import Any, Callable, Dict

import numpy as np


def apply_random_gaussian_noise(img: np.ndarray, noise_cfg: Dict[str, Any]) -> np.ndarray:
    params = noise_cfg.get("random_gaussian_noise", {})
    if not noise_cfg.get("enabled", True) or not params.get("enabled", True):
        return img
    if np.random.random() >= params.get("probability", 1.0):
        return img
    noise = np.random.normal(
        params.get("mean", 0.0),
        params.get("std", 1.0),
        img.shape,
    ).astype(np.float32)
    return img + noise


def _add_depth_artifacts_np(
    img: np.ndarray,
    artifacts_prob: float,
    height_mean: float,
    height_std: float,
    width_mean: float,
    width_std: float,
    noise_value: float = 0.0,
) -> np.ndarray:
    H, W = img.shape
    img = img.copy()

    trigger_mask = (
        (np.random.random(img.shape).astype(np.float32) < artifacts_prob)
        & (img > 0.0)
    )
    coords = np.argwhere(trigger_mask)

    if len(coords) == 0:
        return img

    for cy, cx in coords:
        h_size = int(np.clip(np.random.normal(height_mean, height_std), 0, H))
        w_size = int(np.clip(np.random.normal(width_mean, width_std), 0, W))
        top = int(np.clip(cy - h_size // 2, 0, H))
        bottom = int(np.clip(cy + h_size // 2, 0, H))
        left = int(np.clip(cx - w_size // 2, 0, W))
        right = int(np.clip(cx + w_size // 2, 0, W))
        img[top:bottom, left:right] = noise_value

    return img


def apply_depth_artifact_noise(img: np.ndarray, noise_cfg: Dict[str, Any]) -> np.ndarray:
    params = noise_cfg.get("depth_artifact_noise", {})
    if not noise_cfg.get("enabled", True) or not params.get("enabled", True):
        return img
    return _add_depth_artifacts_np(
        img,
        artifacts_prob=params.get("artifacts_prob", 0.0001),
        height_mean=params.get("artifact_height_mean", 2.0),
        height_std=params.get("artifact_height_std", 0.5),
        width_mean=params.get("artifact_width_mean", 2.0),
        width_std=params.get("artifact_width_std", 0.5),
        noise_value=params.get("artifact_noise_value", 0.0),
    )


def apply_range_based_gaussian_noise(img: np.ndarray, noise_cfg: Dict[str, Any]) -> np.ndarray:
    params = noise_cfg.get("range_based_gaussian_noise", {})
    if not noise_cfg.get("enabled", True) or not params.get("enabled", True):
        return img
    noise = np.random.normal(0.0, params.get("noise_std", 0.02), img.shape).astype(np.float32)
    mask = (img >= params.get("min_value", 0.2)) & (img <= params.get("max_value", 1.5))
    img = img.copy()
    img[mask] += noise[mask]
    return img


def apply_depth_stereo_noise(img: np.ndarray, noise_cfg: Dict[str, Any]) -> np.ndarray:
    params = noise_cfg.get("depth_stereo_noise", {})
    if not noise_cfg.get("enabled", True) or not params.get("enabled", True):
        return img
    img = img.copy()
    H, W = img.shape

    stereo_far_distance = params.get("stereo_far_distance", 2.0)
    stereo_min_distance = params.get("stereo_min_distance", 0.12)

    far_mask = img > stereo_far_distance
    too_close_mask = img < stereo_min_distance
    near_mask = (~far_mask) & (~too_close_mask)

    far_noise = np.random.uniform(0.0, params.get("stereo_far_noise_std", 0.08), img.shape).astype(np.float32)
    img += far_noise * far_mask.astype(np.float32)

    near_noise = np.random.uniform(0.0, params.get("stereo_near_noise_std", 0.02), img.shape).astype(np.float32)
    img += near_noise * near_mask.astype(np.float32)

    vertical_close_ratio = too_close_mask.sum(axis=0, keepdims=True) / H
    vertical_block_mask = vertical_close_ratio > 0.6
    full_block_mask = vertical_block_mask & too_close_mask
    half_block_mask = (~vertical_block_mask) & too_close_mask

    full_values = params.get("stereo_full_block_values", [0.0, 0.25, 0.5, 1.0, 3.0])
    for pixel_value in _random.sample(full_values, len(full_values)):
        artifacts_buffer = np.ones_like(img)
        artifacts_buffer = _add_depth_artifacts_np(
            artifacts_buffer,
            artifacts_prob=params.get("stereo_full_block_artifacts_prob", 0.001),
            height_mean=params.get("stereo_full_block_height_mean", 62.0),
            height_std=params.get("stereo_full_block_height_std", 1.5),
            width_mean=params.get("stereo_full_block_width_mean", 3.0),
            width_std=params.get("stereo_full_block_width_std", 0.01),
            noise_value=0.0,
        )
        img[full_block_mask] = ((1.0 - artifacts_buffer) * pixel_value)[full_block_mask]

    half_block_spark = (
        np.random.uniform(0.0, 1.0, img.shape).astype(np.float32)
        < params.get("stereo_half_block_spark_prob", 0.02)
    )
    img[half_block_mask] = (
        half_block_spark.astype(np.float32) * params.get("stereo_half_block_value", 3.0)
    )[half_block_mask]

    return img


NOISE_FUNCTIONS: Dict[str, Callable] = {
    "random_gaussian_noise": apply_random_gaussian_noise,
    "depth_artifact_noise": apply_depth_artifact_noise,
    "range_based_gaussian_noise": apply_range_based_gaussian_noise,
    "depth_stereo_noise": apply_depth_stereo_noise,
}


def apply_noise_pipeline(img: np.ndarray, noise_cfg: Dict[str, Any]) -> np.ndarray:
    if not noise_cfg.get("enabled", True):
        return img
    for step in noise_cfg.get("order", []):
        fn = NOISE_FUNCTIONS.get(step)
        if fn is None:
            continue
        img = fn(img, noise_cfg)
    return img
