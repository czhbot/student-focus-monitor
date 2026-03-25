"""
模型下载脚本
自动下载项目所需的模型文件
"""

import os
import sys
import subprocess
from pathlib import Path


def get_project_root():
    return Path(__file__).parent.resolve()


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
    print("下载 SixDRepNet 头部姿态模型")
    print("=" * 50)
    
    root = get_project_root()
    xml_path = root / "sixdrepnet_openvino.xml"
    bin_path = root / "sixdrepnet_openvino.bin"
    
    if xml_path.exists() and bin_path.exists():
        print(f"OpenVINO 模型已存在:")
        print(f"  - {xml_path}")
        print(f"  - {bin_path}")
        return True
    
    print("正在安装 SixDRepNet 包 (pip install sixdrepnet)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sixdrepnet", "-q"])
        print("SixDRepNet 包安装成功!")
        
        print("\n正在下载模型权重 (首次使用会自动下载)...")
        from sixdrepnet import SixDRepNet
        model = SixDRepNet()
        print("SixDRepNet 模型准备完成!")
        
        print("\n" + "-" * 50)
        print("注意: 当前使用 PyTorch 版本的 SixDRepNet")
        print("如需 OpenVINO 加速版本，请按以下步骤操作:")
        print("-" * 50)
        print("\n方法一: 从源码转换 (推荐)")
        print("  1. git clone https://github.com/thohemp/6DRepNet.git")
        print("  2. cd 6DRepNet")
        print("  3. pip install -r requirements.txt")
        print("  4. 下载预训练权重: 6DRepNet_300W_LP_AFLW2000.pth")
        print("     https://drive.google.com/drive/folders/1V1pCV0BEW3mD-B9MogGrz_P91UhTtuE_")
        print("  5. 转换为 ONNX: python export_onnx.py")
        print("  6. 转换为 OpenVINO: mo --input_model sixdrepnet.onnx")
        print("  7. 将生成的 .xml 和 .bin 文件复制到项目根目录")
        
        print("\n方法二: 直接使用 PyTorch 版本")
        print("  程序会自动回退到 PyTorch 推理，无需额外操作")
        
        return True
        
    except Exception as e:
        print(f"安装失败: {e}")
        print("\n请手动安装:")
        print("  pip install sixdrepnet")
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
        model_path = root / "yolo11s-pose.pt"
        model = YOLO(str(model_path))
        model.export(format="openvino", half=True, project=str(root))
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
    
    optional = [
        ("sixdrepnet", "头部姿态估计 (可选)"),
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
    
    for package, desc in optional:
        try:
            __import__(package)
            print(f"  ✓ {package} ({desc})")
        except ImportError:
            print(f"  ○ {package} ({desc}) - 未安装，将自动安装")
    
    if missing:
        print(f"\n缺少必需依赖包: {', '.join(missing)}")
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
    print(f"  SixDRepNet:  pip 包形式安装 (或 OpenVINO 版本)")
    
    print("\n准备就绪后，运行: python focus_monitor.py")


if __name__ == "__main__":
    main()
