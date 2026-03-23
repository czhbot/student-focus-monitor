"""
模型下载脚本
自动下载项目所需的模型文件
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.resolve()


def download_file(url: str, save_path: str, desc: str = None):
    if desc:
        print(f"正在下载: {desc}")
    print(f"URL: {url}")
    print(f"保存到: {save_path}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        percent = min(percent, 100)
        sys.stdout.write(f"\r下载进度: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, save_path, progress_hook)
        print("\n下载完成!")
        return True
    except Exception as e:
        print(f"\n下载失败: {e}")
        return False


def download_yolo_model():
    print("\n" + "=" * 50)
    print("下载 YOLO11-Pose 模型")
    print("=" * 50)
    
    root = get_project_root()
    model_path = root / "yolo11s-pose.pt"
    
    if model_path.exists():
        print(f"模型已存在: {model_path}")
        return True
    
    print("正在通过 Ultralytics 下载 YOLO11-Pose 模型...")
    print("(首次运行时会自动下载，或手动下载)")
    
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11s-pose.pt")
        print("YOLO11-Pose 模型下载成功!")
        return True
    except Exception as e:
        print(f"自动下载失败: {e}")
        print("\n请手动下载模型:")
        print("1. 访问 https://github.com/ultralytics/assets/releases")
        print("2. 下载 yolo11s-pose.pt")
        print(f"3. 放置到: {model_path}")
        return False


def download_sixdrepnet_model():
    print("\n" + "=" * 50)
    print("下载 SixDRepNet OpenVINO 模型")
    print("=" * 50)
    
    root = get_project_root()
    xml_path = root / "sixdrepnet_openvino.xml"
    bin_path = root / "sixdrepnet_openvino.bin"
    
    if xml_path.exists() and bin_path.exists():
        print(f"模型已存在:")
        print(f"  - {xml_path}")
        print(f"  - {bin_path}")
        return True
    
    print("SixDRepNet OpenVINO 模型需要手动准备:")
    print("\n方法一: 从原始模型转换")
    print("1. 克隆 SixDRepNet 仓库:")
    print("   git clone https://github.com/thohemp/6DRepNet")
    print("2. 下载预训练权重:")
    print("   https://drive.google.com/drive/folders/1iD3wJXqH6k8Gg5JZz5cQz8wJm8s9tNvO")
    print("3. 使用 OpenVINO 转换模型:")
    print("   pip install openvino-dev")
    print("   mo --input_model sixdrepnet.onnx --output_dir .")
    
    print("\n方法二: 使用 PyTorch 模型 (备选方案)")
    print("项目代码已内置回退机制，如果没有 OpenVINO 模型，")
    print("会尝试使用 PyTorch 模型进行推理。")
    
    return False


def check_openvino_model():
    print("\n" + "=" * 50)
    print("检查 OpenVINO 模型导出")
    print("=" * 50)
    
    root = get_project_root()
    ov_model_dir = root / "yolo11s-pose_openvino_model"
    
    if ov_model_dir.exists():
        print(f"OpenVINO 模型已存在: {ov_model_dir}")
        return True
    
    print("正在导出 YOLO11-Pose 到 OpenVINO 格式...")
    
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11s-pose.pt")
        model.export(format="openvino", half=True)
        print("OpenVINO 模型导出成功!")
        return True
    except Exception as e:
        print(f"导出失败: {e}")
        print("程序将在运行时自动导出")
        return False


def check_dependencies():
    print("\n" + "=" * 50)
    print("检查依赖包")
    print("=" * 50)
    
    required = [
        ("ultralytics", "YOLO 模型框架"),
        ("cv2", "OpenCV 图像处理"),
        ("numpy", "数值计算"),
        ("openvino", "OpenVINO 推理引擎"),
        ("flask", "Web 框架"),
    ]
    
    missing = []
    for package, desc in required:
        try:
            if package == "cv2":
                import cv2
            else:
                __import__(package)
            print(f"  ✓ {package} ({desc})")
        except ImportError:
            print(f"  ✗ {package} ({desc}) - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True


def main():
    print("=" * 60)
    print("学生课堂专注度监控系统 - 模型下载工具")
    print("=" * 60)
    
    check_dependencies()
    
    yolo_ok = download_yolo_model()
    sixdrepnet_ok = download_sixdrepnet_model()
    
    if yolo_ok:
        check_openvino_model()
    
    print("\n" + "=" * 60)
    print("检查完成!")
    print("=" * 60)
    
    root = get_project_root()
    print("\n模型文件位置:")
    print(f"  YOLO11-Pose: {root / 'yolo11s-pose.pt'}")
    print(f"  OpenVINO:    {root / 'yolo11s-pose_openvino_model/'}")
    print(f"  SixDRepNet:  {root / 'sixdrepnet_openvino.xml'}")
    
    print("\n如果模型文件缺失，请按照上述说明手动下载。")
    print("准备就绪后，运行: python focus_monitor.py")


if __name__ == "__main__":
    main()
