# 学生课堂专注度监控系统

基于 YOLO11-Pose + SixDRepNet 的实时学生课堂专注度监控系统，集成多种性能优化策略，实现高帧率、低延迟的实时监测。

## 项目特性

- 实时检测学生专注状态（专注听讲、专注阅读、低头、东张西望、睡觉等）
- 基于 ByteTrack 的多目标追踪，为每个学生分配唯一 ID
- 头部姿态估计（Pitch/Yaw/Roll）判断视线方向
- Web 端实时监控界面，支持 MJPEG 视频流和 WebSocket 数据推送
- 移动端 App（uni-app），支持配置服务器、实时监控数据展示
- 课堂专注度统计与报告生成

## 系统架构

```
摄像头 → 帧队列 → YOLO11-Pose检测 → ByteTrack追踪 → 头部姿态估计 → 专注度评估 → 状态缓存 → 结果展示
```

## 目录结构

```
├── app/                          # 移动端 App (uni-app)
│   ├── src/
│   │   ├── pages/
│   │   │   ├── index/           # 监控主页
│   │   │   └── config/          # 服务器配置页
│   │   ├── App.vue              # 应用入口
│   │   ├── main.js              # 主入口
│   │   ├── pages.json           # 页面配置
│   │   └── uni.scss             # 全局样式
│   └── static/                  # 静态资源
├── web/                         # Web 服务端
│   ├── app.py                  # Flask Web 服务
│   ├── static/                 # 静态资源
│   └── templates/               # HTML 模板
├── focus_monitor.py            # 核心检测模块
├── download_models.py          # 模型下载脚本
├── requirements.txt            # Python 依赖
└── README.md
```

## 核心优化策略

### 1. OpenVINO 推理加速

- **模型格式**: FP16 半精度量化，减少模型体积和内存占用
- **动态 Batch**: 支持动态批次推理，适配不同数量的检测目标
- **多核 CPU 优化**:
  - 设置 `INFERENCE_NUM_THREADS` 充分利用多核 CPU
  - 使用 `AFFINITY: CORE` 绑定核心减少上下文切换
  - `PERFORMANCE_HINT: LATENCY` 优先低延迟模式

```python
core.set_property("CPU", {
    "NUM_STREAMS": "1",
    "AFFINITY": "CORE",
    "INFERENCE_NUM_THREADS": str(num_threads),
    "PERFORMANCE_HINT": "LATENCY"
})
```

### 2. 批量推理优化

将多个学生的头部姿态估计合并为单次批量推理，显著减少推理调用次数：

```python
def estimate_pose_batch(self, frame, bboxes):
    batch_input = np.stack(batch_tensors, axis=0)
    self.infer_request.infer({self.input_name: batch_input})
```

**效果**: 当检测到 N 个学生时，从 N 次推理减少到 1 次批量推理。

### 3. 跳帧策略 (Frame Skipping)

姿态估计采用跳帧策略，每隔 N 帧才执行一次完整的姿态估计：

```python
pose_skip_frames = 7  # 每7帧执行一次姿态估计
```

**效果**: 姿态估计计算量减少约 85%，同时通过角度缓存保持状态连续性。

### 4. 状态缓存池 (StateBuffer)

使用滑动窗口平滑状态判定，消除状态闪烁：

```python
class StateBuffer:
    def __init__(self, window_size: int = 30):
        self.state_history: deque = deque(maxlen=window_size)
```

**功能**:
- 30 帧滑动窗口平滑状态判定
- 角度数据平滑处理
- 专注率统计
- 状态持续时间追踪（检测睡觉等长时间异常状态）

### 5. 多线程异步架构

Web 服务采用双线程异步架构，分离视频流和 AI 检测：

```
┌─────────────────┐     ┌─────────────────┐
│  视频流线程      │     │  AI检测线程      │
│  30 FPS         │     │  5 FPS          │
│  (MJPEG推流)    │     │  (推理+追踪)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
              ┌─────────────┐
              │  共享帧缓冲  │
              └─────────────┘
```

**效果**: 视频流保持流畅 30 FPS，AI 检测独立运行不影响用户体验。

### 6. 摄像头多线程读取

独立线程持续读取摄像头帧，避免主线程阻塞：

```python
class CameraThread:
    def __init__(self, camera_id=0, width=1280, height=720, fps=60):
        self.frame_queue = Queue(maxsize=2)  # 限制队列大小减少延迟
```

### 7. 纯 NumPy 预处理

头部姿态估计的图像预处理使用纯 NumPy 实现，避免 PyTorch/torchvision 依赖：

```python
def _preprocess_head(self, head_img):
    head_rgb = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
    head_resized = cv2.resize(head_rgb, (224, 224))
    head_normalized = head_resized.astype(np.float32) / 255.0
    head_normalized = (head_normalized - self.mean) / self.std
    return np.transpose(head_normalized, (2, 0, 1))
```

## 专注度状态判定逻辑

基于头部姿态角度（Pitch 俯仰角、Yaw 偏航角）的判定逻辑树：

| 状态 | 条件 | 说明 |
|------|------|------|
| 专注听讲 | \|Yaw\| ≤ 30°, \|Pitch\| ≤ 20° | 正视前方 |
| 专注阅读 | \|Yaw\| ≤ 45°, -30° < Pitch ≤ -20° | 低头看书 |
| 低头 | Pitch < -30° | 头部过低 |
| 东张西望 | \|Yaw\| > 45° | 视线偏离 |
| 睡觉 | 长时间低头/趴桌 | 持续时间超过阈值 |

## 安装与运行

### 环境要求

**后端 (Python)**
- Python 3.8+
- OpenVINO 2023.0+
- 摄像头设备
- CUDA (可选，用于 GPU 加速)

**前端 (移动端 App)**
- Node.js 18.x LTS 或 20.x LTS
- HBuilderX (推荐) 或 VS Code + uni-app 插件

### 快速开始

#### 1. 后端安装

```bash
# 克隆仓库
git clone https://github.com/czhbot/student-focus-monitor.git
cd student-focus-monitor

# 创建虚拟环境 (推荐)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 下载模型
python download_models.py

# 运行后端服务
cd web
python app.py
```

访问 `http://localhost:5000` 查看 Web 监控界面。

#### 2. 移动端 App 安装

```bash
# 进入 app 目录
cd app

# 安装依赖
npm install

# 运行开发服务器
npm run dev:h5
```

或使用 HBuilderX 打开 `app` 目录，选择运行到浏览器。

**App 开发说明：**
- 默认连接 `http://localhost:5000`
- 可在"设置"页面修改服务器 IP 地址
- 支持 H5、微信小程序、App 等多端运行

### 模型文件说明

项目需要以下模型文件，`download_models.py` 会自动处理：

| 模型 | 说明 | 自动下载 |
|------|------|---------|
| `yolo11s-pose.pt` | YOLO11 姿态估计模型 | ✓ |
| `yolo11s-pose_openvino_model/` | OpenVINO 导出模型 | ✓ (首次运行自动导出) |
| `sixdrepnet` | SixDRepNet 头部姿态模型 | ✓ (pip 安装) |

> **注意**: SixDRepNet 通过 `pip install sixdrepnet` 自动安装，模型权重会在首次使用时自动下载。如需 OpenVINO 加速版本，可参考 `download_models.py` 中的说明手动转换。

### 运行命令行版本

```bash
python focus_monitor.py
```

**快捷键**:
- `q` - 退出程序
- `s` - 保存截图
- `d` - 打印详细延迟统计
- `c` - 打印学生专注度统计

### 运行 Web 版本

```bash
cd web
python app.py
```

访问 `http://localhost:5000` 查看监控界面。

## 技术栈

**后端**
- **目标检测**: YOLO11-Pose (Ultralytics)
- **推理引擎**: OpenVINO
- **目标追踪**: ByteTrack
- **头部姿态**: SixDRepNet
- **Web 框架**: Flask + WebSocket
- **视频流**: MJPEG
- **前端样式**: Tailwind CSS (本地)
- **图表库**: Chart.js (BootCDN)
- **图标库**: Font Awesome 5 (BootCDN)

**移动端 App**
- **框架**: uni-app (Vue 3)
- **编译器**: Vite 5.x
- **样式**: SCSS + 液态毛玻璃效果

## 依赖版本

### Python 依赖

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
timm>=0.9.0
openvino>=2023.0.0
flask>=2.3.0
flask-sock>=0.6.0
simple-websocket>=1.0.0,<2.0.0
sixdrepnet
```

### Node.js 依赖

```json
{
  "@dcloudio/uni-app": "3.0.0-alpha-5000420260319001",
  "vue": "3.4.21",
  "vite": "5.2.8",
  "typescript": "^5.4.5"
}
```

## 许可证

MIT License
