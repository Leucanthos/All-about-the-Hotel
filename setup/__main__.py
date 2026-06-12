"""宿说 · 环境配置流水线.

用法:
    python setup.py check      # Step 1: 环境检查
    python setup.py model      # Step 2: 下载 Chinese-CLIP
    python setup.py index      # Step 3: 构建向量索引
    python setup.py gpu        # Step 4: OpenVINO GPU 加速 (可选)
    python setup.py verify     # Step 5: 端到端验证
    python setup.py all        # 全部执行
"""
import sys, os, argparse

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .check import python as _py, deps as _deps, gpu as _gpu, data as _data
from .model import download as _download_model
from .index import build as _build_index
from .gpu import export as _export_gpu
from .verify import run as _verify


def main():
    parser = argparse.ArgumentParser(
        description="宿说 — 环境配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python setup.py check\n  python setup.py index --images 500\n  python setup.py all",
    )
    sub = parser.add_subparsers(dest="step")

    # Step 1: check
    p = sub.add_parser("check", help="环境检查 (Python / 依赖 / GPU / 数据)")
    p.set_defaults(func=cmd_check)

    # Step 2: model
    p = sub.add_parser("model", help="下载 Chinese-CLIP 模型")

    # Step 3: index
    p = sub.add_parser("index", help="下载图片 + 构建 NumPy 向量索引")
    p.add_argument("--images", type=int, default=500)
    p.add_argument("--texts", type=int, default=1000)

    # Step 4: gpu
    p = sub.add_parser("gpu", help="导出 OpenVINO GPU 模型 (Intel Arc/iGPU)")

    # Step 5: verify
    p = sub.add_parser("verify", help="端到端验证 (prompt2image + image2text)")

    # all
    p = sub.add_parser("all", help="完整安装 (check → model → index → verify)")
    p.add_argument("--images", type=int, default=500)
    p.add_argument("--texts", type=int, default=1000)

    args = parser.parse_args()
    step = args.step or "all"

    print("=" * 50)
    print(f"宿说 · 环境配置")
    print("=" * 50)

    if step == "check":
        cmd_check()
    elif step == "model":
        cmd_model()
    elif step == "index":
        cmd_index(args)
    elif step == "gpu":
        cmd_gpu()
    elif step == "verify":
        cmd_verify()
    elif step == "all":
        cmd_all(args)
    else:
        parser.print_help()


# ── Command handlers ──

def cmd_check():
    if not _py(): sys.exit(1)
    gpu_type = _gpu()
    if not _deps(): sys.exit(1)
    if not _data(_PROJ): sys.exit(1)
    print("\n[OK] Environment check complete.")

def cmd_model():
    _download_model()

def cmd_index(args):
    if not _data(_PROJ): sys.exit(1)
    _build_index(_PROJ, args.images, args.texts)
    _verify(_PROJ)

def cmd_gpu():
    _export_gpu(_PROJ)

def cmd_verify():
    _verify(_PROJ)

def cmd_all(args):
    # Step 1: check
    print("[1/5] Environment check")
    if not _py(): sys.exit(1)
    gpu_type = _gpu()
    if not _deps(): sys.exit(1)
    if not _data(_PROJ): sys.exit(1)

    # Step 2: model (skip if cached)
    print("[2/5] Model")
    if not _has_model_cache():
        _download_model()
    else:
        print("[INFO] Chinese-CLIP already cached, skipping download")

    # Step 3: index (skip if exists)
    print("[3/5] Index")
    if not os.path.exists(os.path.join(_PROJ, "data", "vectors")):
        _build_index(_PROJ, args.images, args.texts)
    else:
        print("[INFO] Vector index exists, skipping build")

    # Step 4: GPU
    if gpu_type == "openvino":
        print("[4/5] GPU")
        _export_gpu(_PROJ)
    else:
        print("[4/5] GPU (skipped — not available)")

    # Step 5: verify
    print("[5/5] Verify")
    _verify(_PROJ)

    print("\nSetup complete. Try: hotel search \"游泳池\" --mm --top 3")


def _has_model_cache() -> bool:
    try:
        from transformers.utils.hub import cached_file
        return cached_file("OFA-Sys/chinese-clip-vit-base-patch16", "config.json") is not None
    except Exception:
        return False


if __name__ == "__main__":
    main()
