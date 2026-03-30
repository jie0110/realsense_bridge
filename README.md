# RealSense Bridge

一个用于 RealSense 深度相机的 ROS 2 桥接模块，支持真实相机和模拟相机两种工作模式，具有灵活的配置系统和可定制的噪声处理管道。

## 功能特性

- **双模式支持**
  - 真实模式：直接从 RealSense D435 或其他 RealSense 相机读取深度数据
  - 模拟模式：从外部点云话题接收并处理深度数据

- **灵活的 YAML 配置系统**
  - 所有参数可通过 `config.yaml` 配置
  - 支持自动从默认配置与用户配置合并
  - 无需修改代码即可调整参数

- **可配置的噪声处理管道**
  - 全局噪声控制开关
  - 支持个别噪声类型的启用/禁用
  - 支持自定义噪声应用顺序
  - 内置四种噪声模型：
    - 随机高斯噪声 (Random Gaussian Noise)
    - 深度 Artifact 噪声 (Depth Artifact Noise)
    - 基于范围的高斯噪声 (Range-based Gaussian Noise)
    - 立体相机深度噪声 (Depth Stereo Noise)

- **深度处理**
  - 裁剪、高斯模糊、归一化
  - Blind spot 处理
  - 失效像素修复 (Inpaint)

- **ROS 2 集成**
  - 发布处理后的深度点云 (`PointCloud2`)
  - 发布处理后的深度图像 (`Image`)

## 系统要求

- Python 3.8+
- ROS 2 (Humble 或以上)
- 若使用真实模式：RealSense 硬件 + librealsense2

## 安装依赖

### 基础依赖

```bash
pip install numpy opencv-python pyyaml rclpy sensor-msgs
```

### 真实模式依赖

```bash
# 安装 librealsense2（Ubuntu）
sudo apt install librealsense2-dev

# 安装 pyrealsense2
pip install pyrealsense2
```

## 项目结构

```
./
├── config.yaml                      # 主配置文件
├── sim_realsense/
│   └── sim_realsense_pub.py        # 模拟 RealSense 发布节点
├── real_realsnese/
│   └── realsense_pub.py            # 真实 RealSense 发布节点
├── main.py                          # 项目入口（可选）
├── pyproject.toml                   # 项目配置
├── README.md                        # 本文件
└── .gitignore                       # Git 忽略文件
```

## 使用指南

### 1. 配置文件说明

编辑 `config.yaml` 进行配置：

```yaml
depth_processor:
  # 图像尺寸
  sim_width: 64           # 处理宽度
  sim_height: 36          # 处理高度
  
  # 裁剪参数 (旋转后坐标空间)
  crop_up: 18
  crop_down: 0
  crop_left: 16
  crop_right: 16
  
  # 高斯模糊
  gaussian_kernel: [3, 3]
  gaussian_sigma: 1.0
  
  # 深度归一化
  depth_min: 0.0
  depth_max: 2.5
  normalize: true
  out_min: 0.0
  out_max: 1.0
  
  # Blind spot（无效区域置 0）
  blind_up: 0
  blind_down: 0
  blind_left: 0
  blind_right: 0
  
  # 真实相机参数
  rs_width: 480           # RealSense 分辨率宽
  rs_height: 270          # RealSense 分辨率高
  rs_fps: 60              # RealSense 帧率
  
  # 噪声处理
  noise:
    enabled: true         # 全局噪声开关
    order:                # 噪声应用顺序
      - random_gaussian_noise
      - depth_artifact_noise
      - range_based_gaussian_noise
      - depth_stereo_noise
    
    # 随机高斯噪声
    random_gaussian_noise:
      enabled: true
      mean: 0.0
      std: 1.0
      probability: 0.5    # 每帧触发概率
    
    # 深度 Artifact 噪声
    depth_artifact_noise:
      enabled: true
      artifacts_prob: 0.0001
      artifact_height_mean: 2.0
      artifact_height_std: 0.5
      artifact_width_mean: 2.0
      artifact_width_std: 0.5
      artifact_noise_value: 0.0
    
    # 基于范围的高斯噪声
    range_based_gaussian_noise:
      enabled: true
      min_value: 0.2
      max_value: 1.5
      noise_std: 0.02
    
    # 立体相机深度噪声
    depth_stereo_noise:
      enabled: true
      stereo_far_distance: 2.0
      stereo_min_distance: 0.12
      stereo_far_noise_std: 0.08
      stereo_near_noise_std: 0.02
      stereo_full_block_artifacts_prob: 0.001
      stereo_full_block_values: [0.0, 0.25, 0.5, 1.0, 3.0]
      stereo_full_block_height_mean: 62.0
      stereo_full_block_height_std: 1.5
      stereo_full_block_width_mean: 3.0
      stereo_full_block_width_std: 0.01
      stereo_half_block_spark_prob: 0.02
      stereo_half_block_value: 3.0
```

### 2. 运行真实 RealSense 节点

```bash
# 对象检查连接的 RealSense 相机
rs-enumerate-devices

# 启动真实 RealSense 发布节点
ros2 run realsense_bridge realsense_pub
```

**发布话题：**
- `/camera/processed_depth_cloud` (PointCloud2) - 处理后的深度点云
- `/camera/processed_image` (Image) - 处理后的深度图像

### 3. 运行模拟 RealSense 节点

```bash
# 启动模拟 RealSense 发布节点
ros2 run realsense_bridge sim_realsense_pub
```

**订阅话题：**
- `/camera/depth` (PointCloud2) - 输入深度点云

**发布话题：**
- `/camera/processed_depth_cloud` (PointCloud2) - 处理后的深度点云
- `/camera/processed_image` (Image) - 处理后的深度图像

## 配置示例

### 场景 1：禁用所有噪声

```yaml
depth_processor:
  noise:
    enabled: false
```

### 场景 2：仅启用特定噪声

```yaml
depth_processor:
  noise:
    enabled: true
    random_gaussian_noise:
      enabled: true
    depth_artifact_noise:
      enabled: false
    range_based_gaussian_noise:
      enabled: false
    depth_stereo_noise:
      enabled: false
```

### 场景 3：改变噪声应用顺序

```yaml
depth_processor:
  noise:
    order:
      - depth_stereo_noise
      - random_gaussian_noise
      - depth_artifact_noise
```

## 常见问题

**Q: 连接不到 RealSense 相机？**
- A: 检查 USB 连接和驱动安装：`rs-enumerate-devices`

**Q: 找不到 config.yaml？**
- A: 确保 `config.yaml` 在项目根目录，发布节点会自动查找

**Q: PyYAML 模块未找到？**
- A: 执行 `pip install pyyaml`

**Q: 想自定义噪声参数？**
- A: 编辑 `config.yaml` 的相应配置项，无需修改代码

## 代码架构

### 核心特性

- `DepthProcessorConfig` 数据类：管理所有配置参数
- `load_yaml_config()` 函数：从 YAML 文件加载配置
- `apply_noise_pipeline()` 函数：按配置顺序应用噪声
- `img_process()` 函数：完整的深度图处理流程

### 噪声函数

每个噪声处理函数都接受 `DepthProcessorConfig` 参数，支持启用/禁用：

- `apply_random_gaussian_noise()` - 随机高斯噪声
- `apply_depth_artifact_noise()` - Artifact 噪声
- `apply_range_based_gaussian_noise()` - 范围基础高斯噪声
- `apply_depth_stereo_noise()` - 立体相机噪声

## 开发指南

### 添加新的噪声模型

1. 在 `DEFAULT_CONFIG` 中添加新的噪声配置
2. 创建新的处理函数，如 `apply_my_noise(img, cfg)`
3. 在 `NOISE_FUNCTIONS` 字典中注册新函数
4. 在 `config.yaml` 的 `noise.order` 中添加新函数名

### 修改处理流程

编辑 `img_process()` 函数改变处理顺序或添加新的处理步骤。

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 相关资源

- [ROS 2 文档](https://docs.ros.org/en/humble/)
- [RealSense SDK 文档](https://github.com/IntelRealSense/librealsense)
- [OpenCV 文档](https://docs.opencv.org/)
