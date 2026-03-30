import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField

try:
    import yaml
except ImportError:
    yaml = None

CONFIG_YAML_PATH = os.path.join(os.path.dirname(__file__), "process_cfg.yaml")
print(f"Loading depth processor config from: {CONFIG_YAML_PATH}")

DEFAULT_CONFIG: Dict[str, Any] = {
    "sim_width": 64,
    "sim_height": 36,
    "crop_up": 18,
    "crop_down": 0,
    "crop_left": 16,
    "crop_right": 16,
    "gaussian_kernel": [3, 3],
    "gaussian_sigma": 1.0,
    "depth_min": 0.0,
    "depth_max": 2.5,
    "normalize": True,
    "out_min": 0.0,
    "out_max": 1.0,
    "blind_up": 0,
    "blind_down": 0,
    "blind_left": 0,
    "blind_right": 0,
    "rs_width": 480,
    "rs_height": 270,
    "rs_fps": 60,
    "noise": {
        "enabled": False,
        "order": [
            "random_gaussian_noise",
            "depth_artifact_noise",
            "range_based_gaussian_noise",
            "depth_stereo_noise",
        ],
        "random_gaussian_noise": {
            "enabled": True,
            "mean": 0.0,
            "std": 1.0,
            "probability": 0.5,
        },
        "depth_artifact_noise": {
            "enabled": True,
            "artifacts_prob": 0.0001,
            "artifact_height_mean": 2.0,
            "artifact_height_std": 0.5,
            "artifact_width_mean": 2.0,
            "artifact_width_std": 0.5,
            "artifact_noise_value": 0.0,
        },
        "range_based_gaussian_noise": {
            "enabled": True,
            "min_value": 0.2,
            "max_value": 1.5,
            "noise_std": 0.02,
        },
        "depth_stereo_noise": {
            "enabled": True,
            "stereo_far_distance": 2.0,
            "stereo_min_distance": 0.12,
            "stereo_far_noise_std": 0.08,
            "stereo_near_noise_std": 0.02,
            "stereo_full_block_artifacts_prob": 0.001,
            "stereo_full_block_values": [0.0, 0.25, 0.5, 1.0, 3.0],
            "stereo_full_block_height_mean": 62.0,
            "stereo_full_block_height_std": 1.5,
            "stereo_full_block_width_mean": 3.0,
            "stereo_full_block_width_std": 0.01,
            "stereo_half_block_spark_prob": 0.02,
            "stereo_half_block_value": 3.0,
        },
    },
}


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = default.copy()
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_yaml_config(path: str = CONFIG_YAML_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load process_cfg.yaml. Install with 'pip install pyyaml'."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


@dataclass
class DepthProcessorConfig:
    sim_width: int = 64
    sim_height: int = 36
    crop_up: int = 18
    crop_down: int = 0
    crop_left: int = 16
    crop_right: int = 16
    gaussian_kernel: List[int] = field(default_factory=lambda: [3, 3])
    gaussian_sigma: float = 1.0
    depth_min: float = 0.0
    depth_max: float = 2.5
    normalize: bool = True
    out_min: float = 0.0
    out_max: float = 1.0
    blind_up: int = 0
    blind_down: int = 0
    blind_left: int = 0
    blind_right: int = 0
    rs_width: int = 480
    rs_height: int = 270
    rs_fps: int = 60
    noise: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG["noise"])

    @property
    def out_height(self) -> int:
        return self.sim_height - self.crop_up - self.crop_down

    @property
    def out_width(self) -> int:
        return self.sim_width - self.crop_left - self.crop_right

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "DepthProcessorConfig":
        merged = _merge_dict(DEFAULT_CONFIG, cfg.get("depth_processor", {}))
        instance = cls()
        for key, value in merged.items():
            if key == "noise":
                instance.noise = value
                continue
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance


def depth_to_pointcloud2(depth: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    total = depth.size
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = total
    msg.is_bigendian = False
    msg.point_step = 4
    msg.row_step = 4 * total
    msg.is_dense = True
    msg.fields = [PointField(name="z", offset=0, datatype=PointField.FLOAT32, count=1)]
    msg.data = depth.flatten().astype(np.float32).tobytes()
    return msg


def img_process(
    img: np.ndarray,
    cfg: DepthProcessorConfig,
    noise_pipeline: Optional[Callable] = None,
) -> np.ndarray:
    img = img.copy()

    h, w = img.shape
    top = cfg.crop_up
    bottom = h - cfg.crop_down if cfg.crop_down > 0 else h
    left = cfg.crop_left
    right = w - cfg.crop_right if cfg.crop_right > 0 else w
    img = img[top:bottom, left:right]

    if noise_pipeline is not None:
        img = noise_pipeline(img, cfg.noise)

    if any([cfg.blind_up, cfg.blind_down, cfg.blind_left, cfg.blind_right]):
        rh, rw = img.shape
        if cfg.blind_up > 0:
            img[:cfg.blind_up, :] = 0.0
        if cfg.blind_down > 0:
            img[rh - cfg.blind_down:, :] = 0.0
        if cfg.blind_left > 0:
            img[:, :cfg.blind_left] = 0.0
        if cfg.blind_right > 0:
            img[:, rw - cfg.blind_right:] = 0.0

    invalid_mask = (img < 0.2).astype(np.uint8)
    if invalid_mask.any():
        img = cv2.inpaint(img, invalid_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    img = cv2.GaussianBlur(img, tuple(cfg.gaussian_kernel), cfg.gaussian_sigma, cfg.gaussian_sigma)

    img = np.clip(img, cfg.depth_min, cfg.depth_max)
    if cfg.normalize and (cfg.depth_max - cfg.depth_min) > 1e-6:
        img = (img - cfg.depth_min) / (cfg.depth_max - cfg.depth_min)
        img = img * (cfg.out_max - cfg.out_min) + cfg.out_min

    return img.astype(np.float32)
