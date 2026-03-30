#!/usr/bin/env python3

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField

try:
    import yaml
except ImportError:
    yaml = None

CONFIG_YAML_PATH = os.path.join(os.path.dirname(__file__), "../config/config.yaml")

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
    "noise": {
        "enabled": True,
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
            "PyYAML is required to load config.yaml. Install with 'pip install pyyaml'."
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


CFG = DepthProcessorConfig.from_dict(load_yaml_config())


def apply_random_gaussian_noise(img: np.ndarray, cfg: DepthProcessorConfig) -> np.ndarray:
    params = cfg.noise.get("random_gaussian_noise", {})
    if not cfg.noise.get("enabled", True) or not params.get("enabled", True):
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


def apply_depth_artifact_noise(img: np.ndarray, cfg: DepthProcessorConfig) -> np.ndarray:
    params = cfg.noise.get("depth_artifact_noise", {})
    if not cfg.noise.get("enabled", True) or not params.get("enabled", True):
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


def apply_range_based_gaussian_noise(img: np.ndarray, cfg: DepthProcessorConfig) -> np.ndarray:
    params = cfg.noise.get("range_based_gaussian_noise", {})
    if not cfg.noise.get("enabled", True) or not params.get("enabled", True):
        return img
    noise = np.random.normal(0.0, params.get("noise_std", 0.02), img.shape).astype(np.float32)
    mask = (img >= params.get("min_value", 0.2)) & (img <= params.get("max_value", 1.5))
    img = img.copy()
    img[mask] += noise[mask]
    return img


def apply_depth_stereo_noise(img: np.ndarray, cfg: DepthProcessorConfig) -> np.ndarray:
    params = cfg.noise.get("depth_stereo_noise", {})
    if not cfg.noise.get("enabled", True) or not params.get("enabled", True):
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

    import random as _random
    for pixel_value in _random.sample(
        params.get("stereo_full_block_values", [0.0, 0.25, 0.5, 1.0, 3.0]),
        len(params.get("stereo_full_block_values", [0.0, 0.25, 0.5, 1.0, 3.0])),
    ):
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


NOISE_FUNCTIONS = {
    "random_gaussian_noise": apply_random_gaussian_noise,
    "depth_artifact_noise": apply_depth_artifact_noise,
    "range_based_gaussian_noise": apply_range_based_gaussian_noise,
    "depth_stereo_noise": apply_depth_stereo_noise,
}


def apply_noise_pipeline(img: np.ndarray, cfg: DepthProcessorConfig) -> np.ndarray:
    if not cfg.noise.get("enabled", True):
        return img
    for step in cfg.noise.get("order", []):
        fn = NOISE_FUNCTIONS.get(step)
        if fn is None:
            continue
        img = fn(img, cfg)
    return img


def img_process(img: np.ndarray) -> np.ndarray:
    img = img.copy()

    h, w = img.shape
    top = CFG.crop_up
    bottom = h - CFG.crop_down if CFG.crop_down > 0 else h
    left = CFG.crop_left
    right = w - CFG.crop_right if CFG.crop_right > 0 else w
    img = img[top:bottom, left:right]

    img = apply_noise_pipeline(img, CFG)

    if any([CFG.blind_up, CFG.blind_down, CFG.blind_left, CFG.blind_right]):
        rh, rw = img.shape
        if CFG.blind_up > 0:
            img[:CFG.blind_up, :] = 0.0
        if CFG.blind_down > 0:
            img[rh - CFG.blind_down :, :] = 0.0
        if CFG.blind_left > 0:
            img[:, :CFG.blind_left] = 0.0
        if CFG.blind_right > 0:
            img[:, rw - CFG.blind_right :] = 0.0

    invalid_mask = (img < 0.2).astype(np.uint8)
    if invalid_mask.any():
        img = cv2.inpaint(img, invalid_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    img = cv2.GaussianBlur(img, tuple(CFG.gaussian_kernel), CFG.gaussian_sigma, CFG.gaussian_sigma)

    img = np.clip(img, CFG.depth_min, CFG.depth_max)
    if CFG.normalize and (CFG.depth_max - CFG.depth_min) > 1e-6:
        img = (img - CFG.depth_min) / (CFG.depth_max - CFG.depth_min)
        img = img * (CFG.out_max - CFG.out_min) + CFG.out_min

    return img.astype(np.float32)


# ------------------------ (the rest of the node code remains unchanged) ------------------------

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


class SimRealsenseNode(Node):

    def __init__(self):
        super().__init__("sim_realsense_node")

        self.img_width = 36
        self.img_height = 64

        self.subscription = self.create_subscription(
            PointCloud2, "/camera/depth", self.callback, 10
        )
        self.point_pub = self.create_publisher(PointCloud2, "/camera/processed_depth_cloud", 10)
        self.img_pub = self.create_publisher(Image, "/camera/processed_image", 10)

        self.get_logger().info("SimRealsenseNode started.")

    def callback(self, msg: PointCloud2) -> None:
        raw_bytes = bytes(msg.data)
        depth_flat = np.frombuffer(raw_bytes, dtype=np.float32)

        expected = self.img_width * self.img_height
        if len(depth_flat) != expected:
            self.get_logger().warn(
                f"Unexpected data length: {len(depth_flat)}, expected {expected}",
                throttle_duration_sec=5.0,
            )
            return

        depth_image = depth_flat.reshape(self.img_height, self.img_width).copy()
        depth_image[depth_image < 0] = 0.0
        depth_image = np.rot90(depth_image, k=1)

        processed = img_process(depth_image)

        pc_msg = depth_to_pointcloud2(processed, msg.header.stamp, msg.header.frame_id)
        self.point_pub.publish(pc_msg)

        img_msg = Image()
        img_msg.header.stamp = msg.header.stamp
        img_msg.header.frame_id = msg.header.frame_id
        img_msg.height = depth_image.shape[0]
        img_msg.width = depth_image.shape[1]
        img_msg.encoding = "32FC1"
        img_msg.is_bigendian = False
        img_msg.step = depth_image.shape[1] * 4
        img_msg.data = depth_image.astype(np.float32).tobytes()
        self.img_pub.publish(img_msg)

        self._visualize(depth_image, processed)

    def _visualize(self, raw_rot: np.ndarray, processed: np.ndarray) -> None:
        scale = 8
        valid = raw_rot > 0
        if valid.any():
            d_min, d_max = raw_rot[valid].min(), raw_rot[valid].max()
            norm_raw = np.zeros_like(raw_rot, dtype=np.float32)
            norm_raw[valid] = (raw_rot[valid] - d_min) / (d_max - d_min + 1e-6)
        else:
            norm_raw = np.zeros_like(raw_rot, dtype=np.float32)

        vis_raw = (norm_raw * 255).astype(np.uint8)
        color_raw = cv2.applyColorMap(vis_raw, cv2.COLORMAP_JET)
        color_raw[~valid] = 0
        rh, rw = raw_rot.shape
        cv2.imshow(
            "raw rot90 (colormap)",
            cv2.resize(color_raw, (rw * scale, rh * scale), interpolation=cv2.INTER_NEAREST),
        )

        vis_proc = (processed * 255).astype(np.uint8)
        color_proc = cv2.applyColorMap(vis_proc, cv2.COLORMAP_JET)
        ph, pw = processed.shape
        cv2.imshow(
            "processed (colormap)",
            cv2.resize(color_proc, (pw * scale, ph * scale), interpolation=cv2.INTER_NEAREST),
        )
        cv2.imshow(
            "processed (gray)",
            cv2.resize(vis_proc, (pw * scale, ph * scale), interpolation=cv2.INTER_NEAREST),
        )
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = SimRealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down.")
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
