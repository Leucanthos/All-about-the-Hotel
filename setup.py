"""
宿说 — 环境配置与数据准备.

用法:
    python setup.py              # 完整安装 (检查环境 + 下载模型 + 构建索引)
    python setup.py --check      # 仅检查环境
    python setup.py --index      # 仅重建索引 (数据已就绪时)
    python setup.py --gpu        # 额外导出 OpenVINO GPU 模型
"""

import sys, os, subprocess, shutil, json, time, warnings

PROJECT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT)

# ═══════════════════════════════════════════════════
# Step 1: 环境检查
# ═══════════════════════════════════════════════════

def check_python():
    v = sys.version_info
    if v < (3, 10):
        print(f"[FAIL] Python {v.major}.{v.minor} — need 3.10+")
        return False
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_deps():
    deps = ["torch", "transformers", "numpy", "pandas", "PIL", "requests", "pyarrow"]
    missing = []
    for d in deps:
        try:
            __import__(d)
        except ImportError:
            missing.append(d)
    if missing:
        print(f"[WARN] Missing: {missing}")
        print(f"  Run: pip install -r requirements.txt")
        return False
    print(f"[OK] All {len(deps)} core dependencies")
    return True


def check_gpu():
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
        core = ov.Core()
        if "GPU" in core.available_devices:
            print(f"[OK] OpenVINO GPU available (inference only)")
            return "openvino"
    except Exception:
        pass

    print("[INFO] No GPU detected — using CPU")
    return "cpu"


def check_data():
    required = [
        "data/hotel_reviews_table.parquet",
        "data/images/img_0.jpg",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(PROJECT, f))]
    if missing:
        print(f"[WARN] Missing data files: {missing}")
        print(f"  Place hotel_reviews_table.parquet in data/")
        print(f"  Run: python vibecoding/src/build_index.py to download images and build index")
        return False
    n_images = len(os.listdir(os.path.join(PROJECT, "data", "images")))
    print(f"[OK] Data found: hotel_reviews_table.parquet + {n_images} images")
    return True


# ═══════════════════════════════════════════════════
# Step 2: 模型下载
# ═══════════════════════════════════════════════════

def download_base_model():
    """下载 Chinese-CLIP 到本地缓存."""
    print("\nDownloading Chinese-CLIP base model (if not cached)...")
    try:
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        model = ChineseCLIPModel.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16")
        processor = ChineseCLIPProcessor.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16")
        print("[OK] Chinese-CLIP model loaded")
        return True
    except Exception as e:
        print(f"[FAIL] Cannot download model: {e}")
        print("  Check network connection to huggingface.co")
        print("  Or set HF_ENDPOINT=https://hf-mirror.com for China mirror")
        return False


# ═══════════════════════════════════════════════════
# Step 3: 索引构建
# ═══════════════════════════════════════════════════

def build_index(n_images=500, n_texts=1000):
    """下载图片并构建 NumPy 向量索引."""
    print(f"\nBuilding index ({n_images} images, {n_texts} texts)...")
    sys.path.insert(0, os.path.join(PROJECT, "vibecoding", "src"))
    from build_index import download_images, build_indices
    import pandas as pd

    df = pd.read_parquet(os.path.join(PROJECT, "data", "hotel_reviews_table.parquet"))
    img_list = download_images(df, target_count=n_images)
    build_indices(img_list, df, text_limit=n_texts)
    print("[OK] Index built")


# ═══════════════════════════════════════════════════
# Step 4: OpenVINO GPU (可选)
# ═══════════════════════════════════════════════════

def export_openvino():
    """导出 OpenVINO IR 模型用于 GPU 推理."""
    print("\nExporting OpenVINO IR models...")
    try:
        import torch, torch.nn as nn
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        import openvino as ov

        model = ChineseCLIPModel.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True).eval()
        processor = ChineseCLIPProcessor.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True)

        ov_dir = os.path.join(PROJECT, "data", "openvino")
        os.makedirs(ov_dir, exist_ok=True)

        # Text encoder
        text_path = os.path.join(ov_dir, "text_encoder.xml")
        if not os.path.exists(text_path):
            class TextEnc(nn.Module):
                def __init__(self, m): super().__init__()
                def forward(self, ids, mask):
                    out = m.text_model(input_ids=ids, attention_mask=mask)
                    return m.text_projection(out.last_hidden_state[:, 0, :])

            te = TextEnc(model).eval()
            ti = processor(text=["test"], return_tensors="pt", padding=True,
                          truncation=True, max_length=77)
            with torch.no_grad():
                traced = torch.jit.trace(te, (ti["input_ids"], ti["attention_mask"]))
            ov_model = ov.convert_model(traced, example_input=(ti["input_ids"],
                                         ti["attention_mask"]))
            ov.save_model(ov_model, text_path)
            print(f"  [OK] text_encoder")

        # Vision encoder
        vision_path = os.path.join(ov_dir, "vision_encoder.xml")
        if not os.path.exists(vision_path):
            class VisEnc(nn.Module):
                def __init__(self, m): super().__init__()
                def forward(self, px):
                    out = m.vision_model(pixel_values=px)
                    return m.visual_projection(out.last_hidden_state[:, 0, :])

            ve = VisEnc(model).eval()
            with torch.no_grad():
                traced = torch.jit.trace(ve, torch.randn(1, 3, 224, 224))
            ov_model = ov.convert_model(traced, example_input=torch.randn(1, 3, 224, 224))
            ov.save_model(ov_model, vision_path)
            print(f"  [OK] vision_encoder")

        print("[OK] OpenVINO models exported (GPU inference ready)")
        return True
    except Exception as e:
        print(f"[WARN] OpenVINO export failed: {e}")
        print("  GPU inference unavailable, will use CPU")
        return False


# ═══════════════════════════════════════════════════
# Step 5: 验证
# ═══════════════════════════════════════════════════

def verify():
    """端到端验证."""
    print("\nVerifying...")
    sys.path.insert(0, os.path.join(PROJECT, "vibecoding", "src"))
    from multimodal_api import MultimodalAPI

    api = MultimodalAPI()
    stats = api.stats
    assert stats["images"] > 0, "No images in index!"
    assert stats["texts"] > 0, "No texts in index!"

    r = api.prompt2image("游泳池干净吗", topK=3)
    assert len(r["images"]) == 3, "prompt2image failed"

    r2 = api.image2text(f"data/images/img_0.jpg", topK=3)
    assert len(r2["texts"]) == 3, "image2text failed"

    print(f"[OK] Verified: {stats['images']} images, {stats['texts']} texts")
    print("[OK] All systems go!")
    return True


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="宿说 — 环境配置")
    parser.add_argument("--check", action="store_true", help="仅检查环境")
    parser.add_argument("--index", action="store_true", help="仅重建索引")
    parser.add_argument("--gpu", action="store_true", help="导出 OpenVINO GPU 模型")
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--texts", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 50)
    print("宿说 — 环境配置")
    print("=" * 50)

    # Always check environment
    if not check_python():
        sys.exit(1)
    gpu_type = check_gpu()

    if args.check:
        check_deps()
        check_data()
        print("\nEnvironment check complete.")
        sys.exit(0)

    if args.index:
        if not check_data():
            sys.exit(1)
        build_index(args.images, args.texts)
        verify()
        sys.exit(0)

    if args.gpu:
        export_openvino()
        sys.exit(0)

    # Full setup
    print("\n[Full Setup]")
    if not check_deps():
        print("Install dependencies first: pip install -r requirements.txt")
        sys.exit(1)

    if not check_data():
        print("\nData files missing. Please ensure:")
        print("  1. data/hotel_reviews_table.parquet exists")
        print("  2. Then run: python setup.py --index")
        sys.exit(1)

    if not os.path.exists(os.path.join(PROJECT, "data", "vectors_np")):
        build_index(args.images, args.texts)
    else:
        print("[INFO] Index already exists, skipping build")

    # GPU export (auto if OpenVINO available)
    if gpu_type == "openvino":
        export_openvino()

    verify()
    print("\nSetup complete. Try: python -c \"from vibecoding.src.multimodal_api import MultimodalAPI; api=MultimodalAPI(); print(api.prompt2image('test', topK=3))\"")

    # 提示: 训练模型
    exp_dir = os.path.join(PROJECT, "data", "experiments", "baseline")
    if not os.path.exists(exp_dir):
        print("\n[Optional] Train retrieval models:")
        print("  python vibecoding/research/finetune.py --experiment baseline --epochs 10")
