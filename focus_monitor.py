"""
学生课堂专注度监控系统 - 第三阶段：专注度追踪
集成功能：
1. ByteTrack 目标追踪 - 获取学生专属 ID
2. StateBuffer 状态缓存池 - 消除状态闪烁
3. FocusEvaluator 状态判定逻辑树 - 专注度评估
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import cv2
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO
from collections import deque
from enum import Enum
from typing import Optional, Dict, List, Tuple


class FocusState(Enum):
    FOCUSED_LISTENING = "Focused (Listening)"
    FOCUSED_READING = "Focused (Reading/Writing)"
    NOT_FOCUSED_HEAD_DOWN = "Not Focused (Head Down)"
    NOT_FOCUSED_LOOKING_AROUND = "Not Focused (Looking Around)"
    NOT_FOCUSED_SLEEPING = "Not Focused (Sleeping)"
    NORMAL = "Normal"
    UNKNOWN = "Unknown"


class LatencyStats:
    """延迟统计类"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.camera_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.head_pose_times = deque(maxlen=window_size)
        self.postprocess_times = deque(maxlen=window_size)
        self.display_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
    
    def add(self, category: str, time_ms: float):
        if category == "camera":
            self.camera_times.append(time_ms)
        elif category == "inference":
            self.inference_times.append(time_ms)
        elif category == "head_pose":
            self.head_pose_times.append(time_ms)
        elif category == "postprocess":
            self.postprocess_times.append(time_ms)
        elif category == "display":
            self.display_times.append(time_ms)
        elif category == "total":
            self.total_times.append(time_ms)
    
    def get_avg(self, category: str) -> float:
        times = getattr(self, f"{category}_times", None)
        if times and len(times) > 0:
            return sum(times) / len(times)
        return 0.0
    
    def get_summary(self) -> dict:
        return {
            "camera": self.get_avg("camera"),
            "inference": self.get_avg("inference"),
            "head_pose": self.get_avg("head_pose"),
            "postprocess": self.get_avg("postprocess"),
            "display": self.get_avg("display"),
            "total": self.get_avg("total")
        }


class CameraThread:
    """多线程摄像头读取器"""
    
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 60):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = Queue(maxsize=2)
        self.running = False
        self.cap = None
        self.thread = None
    
    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")
        
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"摄像头实际参数: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return self
    
    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
    
    def read(self):
        return self.frame_queue.get()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()


class StateBuffer:
    """状态缓存池 - 消除状态闪烁，追踪状态持续时间"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.state_history: deque = deque(maxlen=window_size)
        self.pitch_history: deque = deque(maxlen=window_size)
        self.yaw_history: deque = deque(maxlen=window_size)
        self.current_state_start_time = None
        self.current_state_duration = 0
        self.current_state = FocusState.UNKNOWN
    
    def add_state(self, state: FocusState, pitch: float = None, yaw: float = None):
        current_time = time.time()
        
        if state != self.current_state:
            if self.current_state != FocusState.UNKNOWN:
                self.current_state_duration = current_time - self.current_state_start_time
            self.current_state = state
            self.current_state_start_time = current_time
        
        self.state_history.append(state)
        if pitch is not None:
            self.pitch_history.append(pitch)
        if yaw is not None:
            self.yaw_history.append(yaw)
    
    def get_smoothed_state(self) -> FocusState:
        if len(self.state_history) == 0:
            return FocusState.UNKNOWN
        
        state_counts: Dict[FocusState, int] = {}
        for state in self.state_history:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        most_common_state = max(state_counts, key=state_counts.get)
        
        current_time = time.time()
        if self.current_state != FocusState.UNKNOWN:
            current_duration = current_time - self.current_state_start_time
            
            if self.current_state == FocusState.FOCUSED_LISTENING and current_duration > 900:
                return FocusState.NOT_FOCUSED_SLEEPING
            elif self.current_state == FocusState.FOCUSED_READING and current_duration > 900:
                return FocusState.NOT_FOCUSED_SLEEPING
            elif self.current_state == FocusState.NOT_FOCUSED_LOOKING_AROUND and current_duration > 420:
                return FocusState.NOT_FOCUSED_SLEEPING
            elif self.current_state == FocusState.NOT_FOCUSED_HEAD_DOWN and current_duration > 600:
                return FocusState.NOT_FOCUSED_SLEEPING
        
        return most_common_state
    
    def get_smoothed_angles(self) -> Tuple[Optional[float], Optional[float]]:
        avg_pitch = None
        avg_yaw = None
        
        if len(self.pitch_history) > 0:
            avg_pitch = sum(self.pitch_history) / len(self.pitch_history)
        if len(self.yaw_history) > 0:
            avg_yaw = sum(self.yaw_history) / len(self.yaw_history)
        
        return avg_pitch, avg_yaw
    
    def get_focus_ratio(self) -> float:
        if len(self.state_history) == 0:
            return 0.0
        
        focused_count = sum(
            1 for state in self.state_history 
            if state in [FocusState.FOCUSED_LISTENING, FocusState.FOCUSED_READING]
        )
        return focused_count / len(self.state_history)
    
    def clear(self):
        self.state_history.clear()
        self.pitch_history.clear()
        self.yaw_history.clear()
        self.current_state = FocusState.UNKNOWN
        self.current_state_start_time = None
        self.current_state_duration = 0


class FocusEvaluator:
    """专注度评估器 - 状态判定逻辑树"""
    
    def __init__(
        self,
        pitch_down_threshold: float = -30.0,
        pitch_sleep_threshold: float = -45.0,
        yaw_looking_threshold: float = 45.0,
        yaw_focused_threshold: float = 30.0,
        pitch_focused_threshold: float = 20.0,
        aspect_ratio_sleep_threshold: float = 1.3
    ):
        self.pitch_down_threshold = pitch_down_threshold
        self.pitch_sleep_threshold = pitch_sleep_threshold
        self.yaw_looking_threshold = yaw_looking_threshold
        self.yaw_focused_threshold = yaw_focused_threshold
        self.pitch_focused_threshold = pitch_focused_threshold
        self.aspect_ratio_sleep_threshold = aspect_ratio_sleep_threshold
    
    def evaluate(
        self, 
        pitch: float, 
        yaw: float, 
        bbox_height: float = None, 
        bbox_width: float = None
    ) -> FocusState:
        if pitch is None or yaw is None:
            return FocusState.UNKNOWN
        
        aspect_ratio = None
        if bbox_height is not None and bbox_width is not None and bbox_width > 0:
            aspect_ratio = bbox_height / bbox_width
        
        if pitch < self.pitch_sleep_threshold:
            if aspect_ratio is not None and aspect_ratio > self.aspect_ratio_sleep_threshold:
                return FocusState.NOT_FOCUSED_SLEEPING
            return FocusState.NOT_FOCUSED_HEAD_DOWN
        
        if pitch < self.pitch_down_threshold:
            return FocusState.NOT_FOCUSED_HEAD_DOWN
        
        if abs(yaw) > self.yaw_looking_threshold:
            return FocusState.NOT_FOCUSED_LOOKING_AROUND
        
        if abs(yaw) <= self.yaw_focused_threshold and abs(pitch) <= self.pitch_focused_threshold:
            return FocusState.FOCUSED_LISTENING
        
        if abs(yaw) <= self.yaw_looking_threshold and abs(pitch) <= self.pitch_down_threshold:
            return FocusState.FOCUSED_READING
        
        return FocusState.NORMAL
    
    def get_state_color(self, state: FocusState) -> Tuple[int, int, int]:
        colors = {
            FocusState.FOCUSED_LISTENING: (0, 255, 0),
            FocusState.FOCUSED_READING: (0, 200, 100),
            FocusState.NOT_FOCUSED_HEAD_DOWN: (0, 165, 255),
            FocusState.NOT_FOCUSED_LOOKING_AROUND: (0, 100, 255),
            FocusState.NOT_FOCUSED_SLEEPING: (0, 0, 255),
            FocusState.NORMAL: (255, 255, 0),
            FocusState.UNKNOWN: (128, 128, 128)
        }
        return colors.get(state, (255, 255, 255))


class StudentTracker:
    """学生追踪管理器 - 管理所有学生的追踪状态，支持跳帧策略"""
    
    def __init__(self, buffer_size: int = 30, pose_skip_frames: int = 7):
        self.buffer_size = buffer_size
        self.pose_skip_frames = pose_skip_frames
        self.students: Dict[int, Dict] = {}
        self.evaluator = FocusEvaluator()
        self.frame_count = 0
    
    def update(
        self, 
        track_id: int, 
        pitch: float, 
        yaw: float, 
        bbox: Tuple[int, int, int, int],
        force_update: bool = False
    ) -> FocusState:
        if track_id not in self.students:
            self.students[track_id] = {
                'buffer': StateBuffer(window_size=self.buffer_size),
                'last_seen': time.time(),
                'total_frames': 0,
                'cached_pitch': None,
                'cached_yaw': None,
                'last_pose_frame': -self.pose_skip_frames
            }
        
        student = self.students[track_id]
        student['last_seen'] = time.time()
        student['total_frames'] += 1
        
        should_update_pose = force_update or (
            self.frame_count - student['last_pose_frame'] >= self.pose_skip_frames
        )
        
        if should_update_pose and pitch is not None and yaw is not None:
            student['cached_pitch'] = pitch
            student['cached_yaw'] = yaw
            student['last_pose_frame'] = self.frame_count
        
        effective_pitch = student['cached_pitch']
        effective_yaw = student['cached_yaw']
        
        if effective_pitch is None or effective_yaw is None:
            return FocusState.UNKNOWN
        
        x_min, y_min, x_max, y_max = bbox
        bbox_height = y_max - y_min
        bbox_width = x_max - x_min
        
        current_state = self.evaluator.evaluate(effective_pitch, effective_yaw, bbox_height, bbox_width)
        
        student['buffer'].add_state(current_state, effective_pitch, effective_yaw)
        
        return student['buffer'].get_smoothed_state()
    
    def get_cached_angles(self, track_id: int) -> Tuple[Optional[float], Optional[float]]:
        if track_id in self.students:
            student = self.students[track_id]
            return student['cached_pitch'], student['cached_yaw']
        return None, None
    
    def increment_frame(self):
        self.frame_count += 1
    
    def should_run_pose_estimation(self, track_id: int) -> bool:
        if track_id not in self.students:
            return True
        student = self.students[track_id]
        return self.frame_count - student['last_pose_frame'] >= self.pose_skip_frames
    
    def get_student_info(self, track_id: int) -> Optional[Dict]:
        return self.students.get(track_id)
    
    def get_smoothed_angles(self, track_id: int) -> Tuple[Optional[float], Optional[float]]:
        if track_id in self.students:
            return self.students[track_id]['buffer'].get_smoothed_angles()
        return None, None
    
    def get_focus_ratio(self, track_id: int) -> float:
        if track_id in self.students:
            return self.students[track_id]['buffer'].get_focus_ratio()
        return 0.0
    
    def cleanup_stale_students(self, timeout: float = 5.0):
        current_time = time.time()
        stale_ids = [
            track_id for track_id, student in self.students.items()
            if current_time - student['last_seen'] > timeout
        ]
        for track_id in stale_ids:
            del self.students[track_id]
    
    def get_all_students(self) -> Dict[int, Dict]:
        return self.students
    
    def get_pose_skip_stats(self) -> Dict:
        total_students = len(self.students)
        students_needing_pose = sum(
            1 for track_id in self.students
            if self.should_run_pose_estimation(track_id)
        )
        return {
            'total_students': total_students,
            'pose_estimations_needed': students_needing_pose,
            'pose_estimations_skipped': total_students - students_needing_pose,
            'skip_ratio': (total_students - students_needing_pose) / total_students if total_students > 0 else 0
        }


class HeadPoseEstimator:
    """头部姿态估计器 - 基于SixDRepNet (OpenVINO)，支持批量推理与动态Batch"""
    
    def __init__(self, model_path: str = None, device: str = "CPU"):
        self.device = device
        self.keypoint_indices = [0, 1, 2, 3, 4]
        self.padding_ratio = 0.3
        
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "sixdrepnet_openvino.xml"
            )
        
        self.model_path = model_path
        self.available = False
        self.model = None
        self.compiled_model = None
        self.infer_request = None
        
        try:
            import openvino as ov
            from openvino.runtime import PartialShape
            
            print(f"头部姿态估计设备: {self.device}")
            print(f"正在加载 SixDRepNet OpenVINO 模型: {model_path}")
            
            core = ov.Core()
            
            cpu_cores = os.cpu_count() or 8
            num_threads = min(cpu_cores, 12)
            
            core.set_property("CPU", {
                "NUM_STREAMS": "1",
                "AFFINITY": "CORE",
                "INFERENCE_NUM_THREADS": str(num_threads),
                "PERFORMANCE_HINT": "LATENCY"
            })
            
            self.model = core.read_model(model_path)
            
            input_name = self.model.inputs[0].any_name
            self.model.reshape({input_name: PartialShape(["?", 3, 224, 224])})
            
            self.compiled_model = core.compile_model(self.model, self.device)
            self.infer_request = self.compiled_model.create_infer_request()
            
            self.input_name = input_name
            self.output_name = self.model.outputs[0].any_name
            
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            print(f"SixDRepNet OpenVINO 模型加载成功!")
            print(f"  输入: {self.input_name}, 形状: {self.model.inputs[0].partial_shape}")
            print(f"  输出: {self.output_name}, 形状: {self.model.outputs[0].partial_shape}")
            self.available = True
            
        except Exception as e:
            print(f"加载 SixDRepNet OpenVINO 失败: {e}")
            self.model = None
            self.compiled_model = None
            self.available = False
    
    def get_head_bbox_from_keypoints(self, keypoints, frame_shape, confidence_threshold=0.5):
        valid_points = []
        
        for idx in self.keypoint_indices:
            x, y, conf = keypoints[idx]
            if conf >= confidence_threshold:
                valid_points.append((x, y))
        
        if len(valid_points) < 2:
            return None
        
        valid_points = np.array(valid_points)
        
        center_x = np.mean(valid_points[:, 0])
        center_y = np.mean(valid_points[:, 1])
        
        distances = np.sqrt((valid_points[:, 0] - center_x)**2 + 
                           (valid_points[:, 1] - center_y)**2)
        radius = np.max(distances) if len(distances) > 0 else 50
        
        radius = max(radius, 30)
        
        padding = radius * self.padding_ratio
        
        height, width = frame_shape[:2]
        
        x_min = max(0, int(center_x - radius - padding))
        y_min = max(0, int(center_y - radius - padding))
        x_max = min(width, int(center_x + radius + padding))
        y_max = min(height, int(center_y + radius + padding))
        
        if x_max - x_min < 20 or y_max - y_min < 20:
            return None
        
        return (x_min, y_min, x_max, y_max)
    
    def _preprocess_head(self, head_img):
        """纯Numpy预处理 - 替代torchvision.transforms"""
        head_rgb = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
        head_resized = cv2.resize(head_rgb, (224, 224))
        head_normalized = head_resized.astype(np.float32) / 255.0
        head_normalized = (head_normalized - self.mean) / self.std
        head_tensor = np.transpose(head_normalized, (2, 0, 1))
        return head_tensor
    
    def estimate_pose(self, frame, bbox):
        if not self.available:
            return None, None, None
        
        x_min, y_min, x_max, y_max = bbox
        
        head_img = frame[y_min:y_max, x_min:x_max, :]
        
        if head_img.size == 0:
            return None, None, None
        
        try:
            head_tensor = self._preprocess_head(head_img)
            batch_input = np.expand_dims(head_tensor, axis=0)
            
            self.infer_request.infer({self.input_name: batch_input})
            output = self.infer_request.get_output_tensor().data
            
            rotation_matrix = output[0]
            pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
            
            return pitch, yaw, roll
        except Exception as e:
            print(f"姿态估计错误: {e}")
            return None, None, None
    
    def _rotation_matrix_to_euler(self, R):
        import math
        
        R = R.reshape(3, 3)
        
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0
        
        pitch = -pitch * 180.0 / math.pi
        yaw = yaw * 180.0 / math.pi
        roll = -roll * 180.0 / math.pi
        
        return pitch, yaw, roll
    
    def estimate_pose_batch(
        self, 
        frame, 
        bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[Optional[float], Optional[float], Optional[float]]]:
        """
        批量头部姿态估计 - OpenVINO动态Batch推理
        
        Parameters:
            frame: 原始帧
            bboxes: 头部边界框列表 [(x1,y1,x2,y2), ...]
        
        Returns:
            [(pitch, yaw, roll), ...] 每个学生的姿态角度
        """
        if not self.available or len(bboxes) == 0:
            return [(None, None, None)] * len(bboxes)
        
        valid_indices = []
        batch_tensors = []
        
        for idx, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            
            head_img = frame[y_min:y_max, x_min:x_max, :]
            
            if head_img.size == 0:
                continue
            
            try:
                head_tensor = self._preprocess_head(head_img)
                valid_indices.append(idx)
                batch_tensors.append(head_tensor)
            except Exception:
                continue
        
        if len(batch_tensors) == 0:
            return [(None, None, None)] * len(bboxes)
        
        try:
            batch_input = np.stack(batch_tensors, axis=0)
            
            self.infer_request.infer({self.input_name: batch_input})
            output = self.infer_request.get_output_tensor().data
            
            results = [(None, None, None)] * len(bboxes)
            for i, valid_idx in enumerate(valid_indices):
                rotation_matrix = output[i]
                pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
                results[valid_idx] = (float(pitch), float(yaw), float(roll))
            
            return results
            
        except Exception as e:
            print(f"批量推理错误: {e}")
            import traceback
            traceback.print_exc()
            return [(None, None, None)] * len(bboxes)
    
    def draw_axis(self, frame, yaw, pitch, roll, center_x, center_y, size=80):
        if not self.available:
            return frame
        
        import math
        
        yaw_rad = yaw * math.pi / 180.0
        pitch_rad = pitch * math.pi / 180.0
        roll_rad = roll * math.pi / 180.0
        
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])
        
        Ry = np.array([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ])
        
        Rz = np.array([
            [math.cos(roll_rad), -math.sin(roll_rad), 0],
            [math.sin(roll_rad), math.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        axes = np.array([
            [size, 0, 0],
            [0, size, 0],
            [0, 0, size]
        ])
        
        axes_2d = (R @ axes.T).T
        
        origin = np.array([center_x, center_y])
        
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        labels = ['X', 'Y', 'Z']
        
        for i, (axis, color, label) in enumerate(zip(axes_2d, colors, labels)):
            end_point = (int(origin[0] + axis[0]), int(origin[1] + axis[1]))
            cv2.line(frame, (int(origin[0]), int(origin[1])), end_point, color, 2)
            cv2.putText(frame, label, end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame


class OpenVINOStudentDetector:
    """OpenVINO多核CPU加速检测器 - 支持ByteTrack追踪"""
    
    def __init__(self, model_name: str = "yolo11s-pose.pt", confidence: float = 0.5):
        import openvino as ov
        import torch
        
        print("=" * 50)
        print("OpenVINO 多核CPU加速模式 + ByteTrack追踪")
        print("=" * 50)
        
        core = ov.Core()
        
        devices = core.available_devices
        print(f"可用设备: {devices}")
        
        cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")
        print(f"CPU: {cpu_name}")
        
        cpu_cores = os.cpu_count() or 8
        num_threads = min(cpu_cores, 12)
        
        core.set_property("CPU", {
            "NUM_STREAMS": "1",
            "AFFINITY": "CORE",
            "INFERENCE_NUM_THREADS": str(num_threads),
            "PERFORMANCE_HINT": "LATENCY"
        })
        
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        
        print(f"CPU核心数: {cpu_cores}")
        print(f"推理线程数: {num_threads}")
        
        self.confidence = confidence
        self.fps_history = []
        self.fps_window = 30
        self.latency_stats = LatencyStats()
        
        ov_model_dir = self._ensure_openvino_model(model_name)
        
        print(f"\n正在加载模型: {ov_model_dir}...")
        self.model = YOLO(ov_model_dir)
        print("模型加载完成!")
    
    def _ensure_openvino_model(self, model_name: str) -> str:
        project_root = os.path.dirname(os.path.abspath(__file__))
        base_name = model_name.replace(".pt", "")
        ov_model_dir = os.path.join(project_root, f"{base_name}_openvino_model")
        
        if os.path.exists(ov_model_dir):
            return ov_model_dir
        
        model_path = os.path.join(project_root, model_name)
        if not os.path.exists(model_path):
            model_path = model_name
        
        print(f"正在导出 OpenVINO 模型: {model_path}...")
        temp_model = YOLO(model_path)
        temp_model.export(format="openvino", half=True, project=project_root)
        
        return ov_model_dir
    
    def calculate_fps(self, elapsed_time: float) -> float:
        if elapsed_time <= 0:
            return 0.0
        current_fps = 1.0 / elapsed_time
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.fps_window:
            self.fps_history.pop(0)
        return sum(self.fps_history) / len(self.fps_history)
    
    def process_frame_with_tracking(self, frame: np.ndarray, persist: bool = True):
        inference_start = time.time()
        
        results = self.model.track(
            source=frame,
            conf=self.confidence,
            verbose=False,
            classes=[0],
            persist=persist,
            tracker="bytetrack.yaml"
        )
        
        inference_time = time.time() - inference_start
        self.latency_stats.add("inference", inference_time * 1000)
        
        return results[0], inference_time
    
    def draw_info(
        self, 
        frame: np.ndarray, 
        fps: float, 
        person_count: int, 
        pose_count: int,
        focus_stats: Dict = None
    ) -> np.ndarray:
        display_start = time.time()
        info_frame = frame.copy()
        
        y_offset = 30
        line_height = 28
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(info_frame, fps_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += line_height
        
        count_text = f"Students: {person_count} | Tracked: {pose_count}"
        cv2.putText(info_frame, count_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += line_height
        
        if focus_stats:
            focused = focus_stats.get('focused', 0)
            total = focus_stats.get('total', 0)
            if total > 0:
                ratio = focused / total * 100
                focus_text = f"Focus Rate: {ratio:.1f}% ({focused}/{total})"
                color = (0, 255, 0) if ratio > 70 else (0, 255, 255) if ratio > 40 else (0, 0, 255)
                cv2.putText(info_frame, focus_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        display_time = time.time() - display_start
        self.latency_stats.add("display", display_time * 1000)
        
        return info_frame


def draw_student_info(
    frame, 
    track_id: int, 
    bbox: Tuple[int, int, int, int],
    focus_state: FocusState,
    evaluator: FocusEvaluator,
    pitch: float = None,
    yaw: float = None,
    focus_ratio: float = None
):
    x_min, y_min, x_max, y_max = bbox
    
    color = evaluator.get_state_color(focus_state)
    
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    id_text = f"ID:{track_id}"
    cv2.putText(frame, id_text, (x_min, y_min - 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    state_text = focus_state.value
    cv2.putText(frame, state_text, (x_min, y_min - 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    if pitch is not None and yaw is not None:
        angle_text = f"P:{pitch:.1f} Y:{yaw:.1f}"
        cv2.putText(frame, angle_text, (x_min, y_min - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if focus_ratio is not None:
        ratio_text = f"Focus:{focus_ratio*100:.0f}%"
        ratio_color = (0, 255, 0) if focus_ratio > 0.7 else (0, 255, 255) if focus_ratio > 0.4 else (0, 0, 255)
        cv2.putText(frame, ratio_text, (x_max - 80, y_min - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ratio_color, 1)
    
    return frame


def main():
    print("=" * 60)
    print("学生课堂专注度监控系统 - 第三阶段")
    print("ByteTrack追踪 + 状态缓存 + 专注度评估 + 跳帧优化 + 批量推理")
    print("=" * 60)
    
    model_name = "yolo11s-pose.pt"
    print(f"\n使用模型: {model_name}")
    print("学生检测后端: OpenVINO FP16 (动态Batch)")
    print("头部姿态后端: OpenVINO (动态Batch)")
    print("追踪算法: ByteTrack")
    
    detector = OpenVINOStudentDetector(model_name=model_name, confidence=0.5)
    
    print("\n初始化头部姿态估计器...")
    head_pose_estimator = HeadPoseEstimator()
    
    buffer_size = 30
    pose_skip_frames = 7
    print(f"\n初始化学生追踪管理器 (缓冲窗口: {buffer_size}帧, 跳帧间隔: {pose_skip_frames}帧)...")
    student_tracker = StudentTracker(buffer_size=buffer_size, pose_skip_frames=pose_skip_frames)
    
    keypoint_confidence_threshold = 0.5
    pose_confidence_threshold = 0.7
    
    print(f"关键点置信度阈值: {keypoint_confidence_threshold}")
    print(f"姿态估计置信度阈值: {pose_confidence_threshold}")
    
    print("\n启动摄像头...")
    camera = CameraThread(camera_id=0, width=1280, height=720, fps=60)
    camera.start()
    
    print(f"摄像头分辨率: {camera.width} x {camera.height}")
    print(f"目标帧率: {camera.fps} fps")
    print("\n按 'q' 键退出程序")
    print("按 's' 键保存当前帧截图")
    print("按 'd' 键打印详细延迟统计")
    print("按 'c' 键打印学生专注度统计")
    print("-" * 60)
    
    frame_count = 0
    screenshot_count = 0
    total_pose_estimations = 0
    total_pose_skipped = 0
    
    try:
        while True:
            total_start = time.time()
            
            camera_start = time.time()
            frame = camera.read()
            camera_time = time.time() - camera_start
            detector.latency_stats.add("camera", camera_time * 1000)
            
            results, inference_time = detector.process_frame_with_tracking(frame, persist=True)
            
            annotated_frame = frame.copy()
            
            student_tracker.increment_frame()
            
            pose_start = time.time()
            pose_count = 0
            focused_count = 0
            frame_pose_estimations = 0
            frame_pose_skipped = 0
            
            if results.boxes is not None and len(results.boxes) > 0:
                person_count = len(results.boxes)
                
                boxes = results.boxes
                keypoints_data = results.keypoints
                
                # === 批量推理优化：先收集所有需要姿态估计的学生 ===
                students_to_process = []
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    track_id = int(box.id.item()) if box.id is not None else i
                    
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
                    bbox = (x_min, y_min, x_max, y_max)
                    
                    if keypoints_data is not None and i < len(keypoints_data):
                        kpts = keypoints_data[i]
                        keypoints = kpts.data[0].cpu().numpy()
                        
                        if len(keypoints) >= 5:
                            head_bbox = head_pose_estimator.get_head_bbox_from_keypoints(
                                keypoints, 
                                frame.shape,
                                confidence_threshold=keypoint_confidence_threshold
                            )
                            
                            if head_bbox is not None:
                                box_conf = np.mean([keypoints[j][2] for j in range(5) if keypoints[j][2] > 0])
                                
                                should_run_pose = student_tracker.should_run_pose_estimation(track_id)
                                
                                if should_run_pose and box_conf >= pose_confidence_threshold:
                                    students_to_process.append({
                                        'idx': i,
                                        'track_id': track_id,
                                        'bbox': bbox,
                                        'head_bbox': head_bbox,
                                        'keypoints': keypoints
                                    })
                                else:
                                    frame_pose_skipped += 1
                
                # === 批量推理：一次性处理所有学生 ===
                if len(students_to_process) > 0:
                    head_bboxes = [s['head_bbox'] for s in students_to_process]
                    pose_results = head_pose_estimator.estimate_pose_batch(frame, head_bboxes)
                    frame_pose_estimations = len(students_to_process)
                    
                    for j, student_info in enumerate(students_to_process):
                        track_id = student_info['track_id']
                        bbox = student_info['bbox']
                        pitch, yaw, roll = pose_results[j]
                        
                        if pitch is not None and yaw is not None:
                            student_tracker.update(track_id, pitch, yaw, bbox, force_update=True)
                
                # === 处理所有学生状态并绘制 ===
                for i in range(len(boxes)):
                    box = boxes[i]
                    track_id = int(box.id.item()) if box.id is not None else i
                    
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].cpu().numpy())
                    bbox = (x_min, y_min, x_max, y_max)
                    
                    focus_state = FocusState.UNKNOWN
                    head_bbox = None
                    
                    if keypoints_data is not None and i < len(keypoints_data):
                        kpts = keypoints_data[i]
                        keypoints = kpts.data[0].cpu().numpy()
                        
                        if len(keypoints) >= 5:
                            head_bbox = head_pose_estimator.get_head_bbox_from_keypoints(
                                keypoints, 
                                frame.shape,
                                confidence_threshold=keypoint_confidence_threshold
                            )
                            
                            if head_bbox is not None:
                                cached_pitch, cached_yaw = student_tracker.get_cached_angles(track_id)
                                
                                if cached_pitch is not None and cached_yaw is not None:
                                    focus_state = student_tracker.update(
                                        track_id, cached_pitch, cached_yaw, bbox, force_update=False
                                    )
                    
                    if focus_state != FocusState.UNKNOWN:
                        pose_count += 1
                        focus_ratio = student_tracker.get_focus_ratio(track_id)
                        smooth_pitch, smooth_yaw = student_tracker.get_smoothed_angles(track_id)
                        
                        if focus_state in [FocusState.FOCUSED_LISTENING, FocusState.FOCUSED_READING]:
                            focused_count += 1
                        
                        if head_bbox is not None:
                            center_x = (head_bbox[0] + head_bbox[2]) // 2
                            center_y = (head_bbox[1] + head_bbox[3]) // 2
                            
                            cached_pitch, cached_yaw = student_tracker.get_cached_angles(track_id)
                            if cached_pitch is not None and cached_yaw is not None:
                                head_pose_estimator.draw_axis(
                                    annotated_frame,
                                    cached_yaw, cached_pitch, 0,
                                    center_x, center_y,
                                    size=min(head_bbox[2] - head_bbox[0], head_bbox[3] - head_bbox[1]) * 0.8
                                )
                        
                        draw_student_info(
                            annotated_frame,
                            track_id,
                            bbox,
                            focus_state,
                            student_tracker.evaluator,
                            smooth_pitch,
                            smooth_yaw,
                            focus_ratio
                        )
                    else:
                        draw_student_info(
                            annotated_frame,
                            track_id,
                            bbox,
                            FocusState.UNKNOWN,
                            student_tracker.evaluator
                        )
            else:
                person_count = 0
            
            student_tracker.cleanup_stale_students(timeout=5.0)
            
            total_pose_estimations += frame_pose_estimations
            total_pose_skipped += frame_pose_skipped
            
            pose_time = time.time() - pose_start
            detector.latency_stats.add("head_pose", pose_time * 1000)
            
            fps = detector.calculate_fps(inference_time)
            
            focus_stats = {
                'focused': focused_count,
                'total': pose_count,
                'pose_estimations': frame_pose_estimations,
                'pose_skipped': frame_pose_skipped
            }
            
            display_frame = detector.draw_info(annotated_frame, fps, person_count, pose_count, focus_stats)
            
            cv2.imshow("Student Attention Monitor - Focus Tracking", display_frame)
            
            total_time = time.time() - total_start
            detector.latency_stats.add("total", total_time * 1000)
            
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n用户请求退出...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_stage3_{screenshot_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"截图已保存: {filename}")
            elif key == ord('d'):
                stats = detector.latency_stats.get_summary()
                print("\n" + "=" * 40)
                print("详细延迟统计 (平均):")
                print("=" * 40)
                print(f"  摄像头读取: {stats['camera']:.2f} ms")
                print(f"  YOLO+追踪:  {stats['inference']:.2f} ms")
                print(f"  头部姿态:   {stats['head_pose']:.2f} ms")
                print(f"  显示绘制:   {stats['display']:.2f} ms")
                print(f"  总耗时:     {stats['total']:.2f} ms")
                print(f"  理论FPS:    {1000/stats['total']:.1f}")
                print("-" * 40)
                print(f"  姿态估计次数: {total_pose_estimations}")
                print(f"  姿态跳过次数: {total_pose_skipped}")
                if total_pose_estimations + total_pose_skipped > 0:
                    skip_ratio = total_pose_skipped / (total_pose_estimations + total_pose_skipped) * 100
                    print(f"  跳帧率: {skip_ratio:.1f}%")
                print("=" * 40)
            elif key == ord('c'):
                students = student_tracker.get_all_students()
                print("\n" + "=" * 50)
                print("学生专注度统计:")
                print("=" * 50)
                for track_id, info in students.items():
                    smooth_state = info['buffer'].get_smoothed_state()
                    focus_ratio = info['buffer'].get_focus_ratio()
                    smooth_pitch, smooth_yaw = info['buffer'].get_smoothed_angles()
                    cached_pitch, cached_yaw = info.get('cached_pitch'), info.get('cached_yaw')
                    print(f"  学生 ID {track_id}:")
                    print(f"    状态: {smooth_state.value}")
                    print(f"    专注率: {focus_ratio*100:.1f}%")
                    if smooth_pitch is not None:
                        print(f"    平均角度: P={smooth_pitch:.1f}° Y={smooth_yaw:.1f}°")
                    if cached_pitch is not None:
                        print(f"    缓存角度: P={cached_pitch:.1f}° Y={cached_yaw:.1f}°")
                    print(f"    追踪帧数: {info['total_frames']}")
                print("=" * 50)
    
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("程序结束")
    print(f"总处理帧数: {frame_count}")
    if detector.fps_history:
        print(f"平均FPS: {sum(detector.fps_history)/len(detector.fps_history):.1f}")
    
    stats = detector.latency_stats.get_summary()
    print("\n最终延迟统计:")
    print(f"  摄像头读取: {stats['camera']:.2f} ms")
    print(f"  YOLO+追踪:  {stats['inference']:.2f} ms")
    print(f"  头部姿态:   {stats['head_pose']:.2f} ms")
    print(f"  显示绘制:   {stats['display']:.2f} ms")
    print(f"  总耗时:     {stats['total']:.2f} ms")
    
    students = student_tracker.get_all_students()
    print(f"\n追踪到的学生总数: {len(students)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
