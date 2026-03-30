#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

from add_noise.add_noise import apply_noise_pipeline
from depth_processor import (
    DepthProcessorConfig,
    depth_to_pointcloud2,
    img_process,
    load_yaml_config,
)

CFG = DepthProcessorConfig.from_dict(load_yaml_config())


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

        noise_enabled = CFG.noise.get("enabled", False)
        self.get_logger().info(
            f"SimRealsenseNode started. Noise pipeline: {'enabled' if noise_enabled else 'disabled'}"
        )

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

        processed = img_process(depth_image, CFG, noise_pipeline=apply_noise_pipeline)

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
