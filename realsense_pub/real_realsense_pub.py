#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2

from depth_processor import (
    DepthProcessorConfig,
    depth_to_pointcloud2,
    img_process,
    load_yaml_config,
)

CFG = DepthProcessorConfig.from_dict(load_yaml_config())


class RealSenseNode(Node):

    def __init__(self):
        super().__init__("realsense_node")

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.depth,
            CFG.rs_width,
            CFG.rs_height,
            rs.format.z16,
            CFG.rs_fps,
        )
        self._pipeline.start(config)

        profile = self._pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        self.get_logger().info(
            f"RealSense started: {CFG.rs_width}x{CFG.rs_height} @ {CFG.rs_fps} Hz, "
            f"depth_scale={self._depth_scale:.6f} m/unit"
        )

        self.point_pub = self.create_publisher(PointCloud2, "/camera/processed_depth_cloud", 10)
        self.img_pub = self.create_publisher(Image, "/camera/processed_image", 10)

        timer_period = 1.0 / CFG.rs_fps
        self._timer = self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info("RealSenseNode started.")

    def _timer_callback(self) -> None:
        frames = self._pipeline.poll_for_frames()
        if not frames:
            return

        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return

        self._process_frame(depth_frame)

    def _process_frame(self, depth_frame) -> None:
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m = depth_raw * self._depth_scale
        depth_m[depth_m < 0] = 0.0

        depth_resized = cv2.resize(
            depth_m,
            (CFG.sim_width, CFG.sim_height),
            interpolation=cv2.INTER_AREA,
        )

        processed = img_process(depth_resized, CFG)

        stamp = self.get_clock().now().to_msg()
        frame_id = "camera_depth_frame"

        self.point_pub.publish(depth_to_pointcloud2(processed, stamp, frame_id))

        img_msg = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = frame_id
        img_msg.height = depth_resized.shape[0]
        img_msg.width = depth_resized.shape[1]
        img_msg.encoding = "32FC1"
        img_msg.is_bigendian = False
        img_msg.step = depth_resized.shape[1] * 4
        img_msg.data = depth_resized.astype(np.float32).tobytes()
        self.img_pub.publish(img_msg)

        self._visualize(depth_resized, processed)

    def _visualize(self, raw_resized: np.ndarray, processed: np.ndarray) -> None:
        scale = 8

        valid = raw_resized > 0
        if valid.any():
            d_min, d_max = raw_resized[valid].min(), raw_resized[valid].max()
            norm_raw = np.zeros_like(raw_resized, dtype=np.float32)
            norm_raw[valid] = (raw_resized[valid] - d_min) / (d_max - d_min + 1e-6)
        else:
            norm_raw = np.zeros_like(raw_resized, dtype=np.float32)

        vis_raw = (norm_raw * 255).astype(np.uint8)
        color_raw = cv2.applyColorMap(vis_raw, cv2.COLORMAP_JET)
        color_raw[~valid] = 0
        rh, rw = raw_resized.shape
        cv2.imshow(
            "img",
            cv2.resize(color_raw, (rw * scale, rh * scale), interpolation=cv2.INTER_NEAREST),
        )

        vis_proc = (processed * 255).astype(np.uint8)
        color_proc = cv2.applyColorMap(vis_proc, cv2.COLORMAP_JET)
        ph, pw = processed.shape
        cv2.imshow(
            "processed",
            cv2.resize(color_proc, (pw * scale, ph * scale), interpolation=cv2.INTER_NEAREST),
        )
        cv2.waitKey(1)

    def destroy_node(self):
        self._pipeline.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = RealSenseNode()
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
