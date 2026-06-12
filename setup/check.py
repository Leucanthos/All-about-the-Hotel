"""环境检查: Python, 依赖, GPU, 数据文件."""
import sys, os


def python():
    v = sys.version_info
    if v < (3, 10):
        print(f"[FAIL] Python {v.major}.{v.minor} — need 3.10+")
        return False
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")
    return True

def deps():
    names = ["torch", "transformers", "numpy", "pandas", "PIL", "requests", "pyarrow"]
    missing = [d for d in names if not _try_import(d)]
    if missing:
        print(f"[WARN] Missing: {missing}")
        print(f"  Run: pip install -r requirements.txt")
        return False
    print(f"[OK] All {len(names)} core dependencies")
    return True

def gpu():
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            print(f"[OK] Intel XPU GPU: {torch.xpu.get_device_name(0)}")
            return "xpu"
        if torch.cuda.is_available():
            print(f"[OK] CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
    except Exception:
        pass
    try:
        import openvino as ov
        if "GPU" in ov.Core().available_devices:
            print(f"[OK] OpenVINO GPU available (inference only)")
            return "openvino"
    except Exception:
        pass
    print("[INFO] No GPU detected — using CPU")
    return "cpu"

def data(project_root: str) -> bool:
    required = ["data/raw/hotel_reviews_table.parquet", "data/images/img_0.jpg"]
    missing = [f for f in required if not os.path.exists(os.path.join(project_root, f))]
    if missing:
        print(f"[WARN] Missing data files: {missing}")
        print(f"  Place hotel_reviews_table.parquet in data/raw/")
        return False
    n = len(os.listdir(os.path.join(project_root, "data", "images")))
    print(f"[OK] Data: hotel_reviews_table.parquet + {n} images")
    return True

def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
