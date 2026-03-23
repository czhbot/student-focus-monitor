"""
学生课堂专注度监控系统 - Web 服务端
Flask + WebSocket + MJPEG 流媒体
双线程异步架构：视频流30FPS + AI检测5FPS
"""

import os
import sys
import time
import json
import cv2
import base64
import threading
import numpy as np
from queue import Queue
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from flask import Flask, render_template, Response, jsonify
from flask_sock import Sock
from simple_websocket import Server, ConnectionClosed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from focus_monitor import (
    OpenVINOStudentDetector,
    HeadPoseEstimator,
    StudentTracker,
    FocusState,
    CameraThread,
    draw_student_info
)


@dataclass
class DetectionResult:
    track_id: int
    bbox: Tuple[int, int, int, int]
    focus_state: FocusState
    smooth_pitch: Optional[float] = None
    smooth_yaw: Optional[float] = None
    focus_ratio: float = 0.0


app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
sock = Sock(app)

ws_clients = set()
ws_lock = threading.Lock()

current_frame = None
frame_counter = 0
raw_frame = None
latest_ai_results = {}
lock = threading.Lock()

latest_stats = {
    'focus_rate': 0,
    'total_students': 0,
    'warning_count': 0,
    'active_count': 0,
    'fps': 0,
    'status_distribution': {
        'focused_listening': 0,
        'focused_reading': 0,
        'not_focused_looking': 0,
        'not_focused_sleeping': 0
    },
    'events': []
}
stats_lock = threading.Lock()

session_focus_history = []
session_low_focus_times = set()
session_max_students = 0

events_history = deque(maxlen=10)

detector = None
head_pose_estimator = None
student_tracker = None
camera = None
stream_thread = None
detection_thread = None
running = False

STREAM_FPS = 30
DETECTION_FPS = 5


def init_system():
    global detector, head_pose_estimator, student_tracker, camera, running

    print("=" * 60)
    print("学生课堂专注度监控系统 - Web 服务端")
    print("双线程异步架构: 视频流30FPS + AI检测5FPS")
    print("=" * 60)

    model_name = "yolo11s-pose.pt"
    print(f"\n使用模型: {model_name}")
    print("推理后端: OpenVINO FP16 (动态Batch)")
    print("头部姿态后端: OpenVINO (动态Batch)")
    print("追踪算法: ByteTrack")

    detector = OpenVINOStudentDetector(model_name=model_name, confidence=0.5)

    print("\n初始化头部姿态估计器...")
    head_pose_estimator = HeadPoseEstimator()

    buffer_size = 30
    pose_skip_frames = 7
    print(f"\n初始化学生追踪管理器 (缓冲窗口: {buffer_size}帧, 跳帧间隔: {pose_skip_frames}帧)...")
    student_tracker = StudentTracker(buffer_size=buffer_size, pose_skip_frames=pose_skip_frames)

    print("\n启动摄像头...")
    camera = CameraThread(camera_id=0, width=1280, height=720, fps=60)
    camera.start()

    print(f"摄像头分辨率: {camera.width} x {camera.height}")
    print(f"目标帧率: {camera.fps} fps")

    running = True
    return True


def add_event(event_type: str, title: str, description: str):
    event = {
        'type': event_type,
        'title': title,
        'description': description,
        'time': datetime.now().strftime('%H:%M:%S')
    }
    events_history.append(event)


def video_stream_loop():
    global current_frame, frame_counter, raw_frame, running

    print("[视频流线程] 启动，目标帧率: 30 FPS")
    frame_interval = 1.0 / STREAM_FPS

    while running:
        try:
            start_time = time.time()

            frame = camera.read()
            if frame is None:
                time.sleep(0.001)
                continue

            with lock:
                raw_frame = frame.copy()
                results_snapshot = list(latest_ai_results.values())

            annotated_frame = frame.copy()

            for result in results_snapshot:
                draw_student_info(
                    annotated_frame,
                    result.track_id,
                    result.bbox,
                    result.focus_state,
                    student_tracker.evaluator,
                    result.smooth_pitch,
                    result.smooth_yaw,
                    result.focus_ratio
                )

            cv2.putText(annotated_frame, f"Stream: {STREAM_FPS}FPS | AI: {DETECTION_FPS}FPS",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            with lock:
                current_frame = annotated_frame.copy()
                frame_counter += 1

            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[视频流线程] 错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.01)


def ai_detection_loop():
    global running, latest_ai_results

    print("[AI检测线程] 启动，目标帧率: 5 FPS")
    frame_interval = 1.0 / DETECTION_FPS

    last_warning_check = 0
    prev_warning_count = 0

    while running:
        try:
            start_time = time.time()

            with lock:
                has_frame = raw_frame is not None
                if has_frame:
                    frame_to_process = raw_frame.copy()

            if not has_frame:
                time.sleep(0.01)
                continue

            results, inference_time = detector.process_frame_with_tracking(frame_to_process, persist=True)

            student_tracker.increment_frame()

            pose_count = 0
            focused_count = 0
            status_counts = {
                FocusState.FOCUSED_LISTENING: 0,
                FocusState.FOCUSED_READING: 0,
                FocusState.NOT_FOCUSED_HEAD_DOWN: 0,
                FocusState.NOT_FOCUSED_LOOKING_AROUND: 0,
                FocusState.NOT_FOCUSED_SLEEPING: 0,
                FocusState.NORMAL: 0,
                FocusState.UNKNOWN: 0,
            }

            new_detection_results = {}

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes
                keypoints_data = results.keypoints

                # === 批量推理优化：先收集所有需要姿态估计的学生 ===
                students_to_process = []
                
                for i in range(len(boxes)):
                    box = boxes[i]

                    if box.id is None:
                        continue
                    track_id = int(box.id.item())

                    bbox_array = box.xyxy[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = np.round(bbox_array).astype(int)
                    bbox = (x_min, y_min, x_max, y_max)

                    if keypoints_data is not None and i < len(keypoints_data):
                        kpts = keypoints_data[i]
                        keypoints = kpts.data[0].cpu().numpy()

                        if len(keypoints) >= 5:
                            head_bbox = head_pose_estimator.get_head_bbox_from_keypoints(
                                keypoints,
                                frame_to_process.shape,
                                confidence_threshold=0.5
                            )

                            if head_bbox is not None:
                                valid_confs = [keypoints[j][2] for j in range(5) if keypoints[j][2] > 0]
                                box_conf = np.mean(valid_confs) if valid_confs else 0.0

                                should_run_pose = student_tracker.should_run_pose_estimation(track_id)

                                if should_run_pose and box_conf >= 0.7:
                                    students_to_process.append({
                                        'track_id': track_id,
                                        'bbox': bbox,
                                        'head_bbox': head_bbox
                                    })

                # === 批量推理：一次性处理所有学生 ===
                if len(students_to_process) > 0:
                    head_bboxes = [s['head_bbox'] for s in students_to_process]
                    pose_results = head_pose_estimator.estimate_pose_batch(frame_to_process, head_bboxes)
                    
                    for j, student_info in enumerate(students_to_process):
                        track_id = student_info['track_id']
                        bbox = student_info['bbox']
                        pitch, yaw, roll = pose_results[j]
                        
                        if pitch is not None and yaw is not None:
                            student_tracker.update(track_id, pitch, yaw, bbox, force_update=True)

                # === 处理所有学生状态并生成结果 ===
                for i in range(len(boxes)):
                    box = boxes[i]

                    if box.id is None:
                        continue
                    track_id = int(box.id.item())

                    bbox_array = box.xyxy[0].cpu().numpy()
                    x_min, y_min, x_max, y_max = np.round(bbox_array).astype(int)
                    bbox = (x_min, y_min, x_max, y_max)

                    focus_state = FocusState.UNKNOWN
                    smooth_pitch = None
                    smooth_yaw = None
                    focus_ratio = 0.0

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

                        status_counts[focus_state] = status_counts.get(focus_state, 0) + 1

                    new_detection_results[track_id] = DetectionResult(
                        track_id=track_id,
                        bbox=bbox,
                        focus_state=focus_state,
                        smooth_pitch=smooth_pitch,
                        smooth_yaw=smooth_yaw,
                        focus_ratio=focus_ratio
                    )

            student_tracker.cleanup_stale_students(timeout=5.0)

            loop_time = time.time() - start_time
            fps = 1.0 / loop_time if loop_time > 0 else 0

            focus_rate = (focused_count / pose_count * 100) if pose_count > 0 else 0
            
            global session_max_students
            if pose_count > session_max_students:
                session_max_students = pose_count
            
            if pose_count > 0:
                session_focus_history.append(focus_rate)
                if focus_rate < 60:
                    time_str = datetime.now().strftime('%H:%M')
                    session_low_focus_times.add(time_str)
            
            warning_count = status_counts.get(FocusState.NOT_FOCUSED_SLEEPING, 0) + \
                           status_counts.get(FocusState.NOT_FOCUSED_HEAD_DOWN, 0)
            active_count = status_counts.get(FocusState.FOCUSED_LISTENING, 0)

            current_time = time.time()
            if current_time - last_warning_check > 5.0:
                if warning_count > prev_warning_count and warning_count > 0:
                    add_event('warning', '检测到注意力下降',
                             f'当前有 {warning_count} 名学生低头或趴桌')
                elif warning_count == 0 and prev_warning_count > 0:
                    add_event('info', '注意力恢复',
                             '所有学生注意力已恢复正常')

                if focus_rate < 60 and pose_count > 0:
                    add_event('warning', '整体专注度偏低',
                             f'当前专注度仅 {focus_rate:.0f}%')
                elif focus_rate > 90 and pose_count > 0:
                    add_event('info', '专注度优秀',
                             f'当前专注度达到 {focus_rate:.0f}%')

                last_warning_check = current_time
                prev_warning_count = warning_count

            with lock:
                latest_ai_results = new_detection_results

            with stats_lock:
                latest_stats['focus_rate'] = focus_rate
                latest_stats['total_students'] = pose_count
                latest_stats['warning_count'] = warning_count
                latest_stats['active_count'] = active_count
                latest_stats['fps'] = fps
                latest_stats['status_distribution'] = {
                    'focused_listening': status_counts.get(FocusState.FOCUSED_LISTENING, 0),
                    'focused_reading': status_counts.get(FocusState.FOCUSED_READING, 0),
                    'not_focused_head_down': status_counts.get(FocusState.NOT_FOCUSED_HEAD_DOWN, 0),
                    'not_focused_looking': status_counts.get(FocusState.NOT_FOCUSED_LOOKING_AROUND, 0),
                    'not_focused_sleeping': status_counts.get(FocusState.NOT_FOCUSED_SLEEPING, 0),
                    'normal': status_counts.get(FocusState.NORMAL, 0),
                }
                latest_stats['stream_status'] = f'Status: Processing {pose_count} targets...'
                latest_stats['events'] = list(events_history)

            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[AI检测线程] 错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)


def generate_mjpeg():
    last_frame_count = -1

    while running:
        with lock:
            frame = current_frame
            current_count = frame_counter

        if frame is not None and current_count != last_frame_count:
            last_frame_count = current_count
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.01)


def broadcast_stats():
    with ws_lock:
        clients = list(ws_clients)

    if not clients:
        return

    with stats_lock:
        data = latest_stats.copy()

    message = json.dumps(data)

    disconnected = []
    for client in clients:
        try:
            client.send(message)
        except ConnectionClosed:
            disconnected.append(client)
        except Exception as e:
            print(f"WebSocket 意外发送错误: {e}")
            disconnected.append(client)

    for client in disconnected:
        with ws_lock:
            ws_clients.discard(client)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@sock.route('/ws')
def websocket_handler(ws: Server):
    global ws_clients

    with ws_lock:
        ws_clients.add(ws)

    print(f"WebSocket 客户端连接，当前连接数: {len(ws_clients)}")

    try:
        while True:
            data = ws.receive()
            if data is None:
                break

            if data == 'ping':
                ws.send('pong')
    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        with ws_lock:
            ws_clients.discard(ws)
        print(f"WebSocket 客户端断开，当前连接数: {len(ws_clients)}")


def websocket_broadcaster():
    while running:
        broadcast_stats()
        time.sleep(0.5)


@app.route('/api/stop', methods=['POST'])
def stop_system():
    global running, camera
    
    avg_focus = sum(session_focus_history) / len(session_focus_history) if session_focus_history else 0
    
    low_focus_list = sorted(list(session_low_focus_times))
    
    report_data = {
        'attendance': session_max_students,
        'avg_focus': round(avg_focus, 1),
        'low_focus_times': low_focus_list
    }
    
    running = False
    if camera:
        camera.stop()
    
    def shutdown():
        time.sleep(2)
        os._exit(0)
    
    threading.Thread(target=shutdown, daemon=True).start()
    
    return jsonify(report_data)


def run_server(host='0.0.0.0', port=5000):
    global stream_thread, detection_thread, running

    if not init_system():
        print("系统初始化失败!")
        return

    stream_thread = threading.Thread(target=video_stream_loop, daemon=True, name="VideoStreamThread")
    stream_thread.start()

    detection_thread = threading.Thread(target=ai_detection_loop, daemon=True, name="AIDetectionThread")
    detection_thread.start()

    broadcaster_thread = threading.Thread(target=websocket_broadcaster, daemon=True, name="WebSocketBroadcaster")
    broadcaster_thread.start()

    print("\n" + "=" * 60)
    print(f"Web 服务启动: http://{host}:{port}")
    print(f"视频流线程: {STREAM_FPS} FPS")
    print(f"AI检测线程: {DETECTION_FPS} FPS")
    print("=" * 60)

    try:
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    finally:
        running = False
        if camera:
            camera.stop()
        print("\n服务已停止")


if __name__ == '__main__':
    run_server()
