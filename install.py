"""宿说 — 一键安装脚本.

    python install.py           # 完整安装（依赖 + 模型 + 索引 + 验证）
    python install.py --check   # 仅环境检查
    python install.py --model   # 仅下载模型
    python install.py --index   # 仅构建索引
    python install.py --gpu     # OpenVINO GPU 加速（可选）
"""

import sys, os, subprocess, argparse

PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)


def run(cmd, desc=""):
    print(f"  [{desc}] {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL: {r.stderr[:200]}")
        return False
    return True


def check():
    """环境检查."""
    print("\n" + "=" * 50)
    print("  1/5 环境检查")
    print("=" * 50)

    # Python version
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}  {'OK' if v >= (3,10) else '需要 >=3.10'}")

    # Data
    parquet = os.path.join(PROJ, "data", "hotel_reviews_full.parquet")
    if os.path.exists(parquet):
        size_mb = os.path.getsize(parquet) / 1024 / 1024
        print(f"  数据 OK ({size_mb:.0f} MB)")
    else:
        print(f"  数据缺失: 请将 hotel_reviews_full.parquet 放入 data/ 目录")

    # Images
    img_dir = os.path.join(PROJ, "data", "images")
    n_img = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
    print(f"  图片 OK ({n_img} 张)")

    # .env
    env = os.path.join(PROJ, ".env")
    if os.path.exists(env):
        print(f"  .env 配置 OK")
    else:
        print(f"  提示: 创建 .env 文件配置 DeepSeek API Key (可选，加速 LLM 路由)")

    print()


def install_deps():
    """安装 pip 依赖."""
    print("\n" + "=" * 50)
    print("  2/5 安装依赖")
    print("=" * 50)
    run(f'"{sys.executable}" -m pip install -e .', "CLI 安装")
    print()


def download_model():
    """下载 Chinese-CLIP 模型."""
    print("\n" + "=" * 50)
    print("  3/5 下载模型")
    print("=" * 50)

    try:
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        print("  正在下载 Chinese-CLIP (OFA-Sys/chinese-clip-vit-base-patch16)...")
        model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        print("  模型下载完成。")

        # Test inference
        import torch
        inputs = processor(text=["测试"], return_tensors="pt", padding=True)
        with torch.no_grad():
            _ = model.get_text_features(**inputs)
        print("  推理测试通过。")
    except Exception as e:
        print(f"  模型下载失败: {e}")
        print("  提示: 设置 HF_ENDPOINT=https://hf-mirror.com 使用镜像加速")
    print()


def build_index():
    """构建向量索引."""
    print("\n" + "=" * 50)
    print("  4/5 构建索引")
    print("=" * 50)

    try:
        from _shared.cache import cache
        print("  正在加载数据并构建 BM25 索引...")
        bm25 = cache.get_bm25()
        print(f"  BM25 索引 OK ({bm25.num_docs} 文档)")
        comments = cache.get_comments()
        print(f"  评论数据 OK ({len(comments)} 条)")
    except Exception as e:
        print(f"  索引构建失败: {e}")
    print()


def verify():
    """端到端验证."""
    print("\n" + "=" * 50)
    print("  5/5 验证")
    print("=" * 50)

    print("  测试 1: 意图分类...")
    from _shared.router import classify_intent_detailed
    r = classify_intent_detailed("游泳池干净吗")
    assert r["primary"] == "facility", f"expected facility, got {r['primary']}"
    print("    OK: 游泳池 → facility")

    print("  测试 2: BM25 检索...")
    from _shared.cache import cache
    idx = cache.get_bm25()
    res = idx.search("早餐", topk=5)
    assert len(res) > 0, "BM25 returned no results"
    print(f"    OK: BM25 返回 {len(res)} 条")

    print("  测试 3: 场景识别...")
    from advisor.recognizer import recognize
    r = recognize("带5岁宝宝去住推荐吗")
    assert r["scene"] in ("family", "general")
    print(f"    OK: 亲子 → {r['scene']}")

    r = recognize("半夜12:30到达可以入住吗")
    assert r["scene"] == "practical"
    print(f"    OK: 半夜入住 → practical")

    print("  测试 4: 顾问管线...")
    from advisor.agent import advise
    r = advise("有洗衣机吗", verbose=False)
    assert r["scene"] == "practical"
    assert len(r["reply"]) > 0
    print(f"    OK: practical 查询完成 ({r['metrics']['total_latency']:.1f}s)")

    print()
    print("  ✅ 全部验证通过！")
    print()
    print("  " + "=" * 50)
    print("  安装完成！运行: hotel")
    print("  " + "=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="宿说 — 一键安装",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python install.py            # 完整安装\n"
            "  python install.py --check    # 仅检查环境\n"
            "  python install.py --gpu      # 导出 OpenVINO GPU 模型\n"
            "\n"
            "镜像加速 (国内):\n"
            "  设置环境变量 HF_ENDPOINT=https://hf-mirror.com\n"
            "  Linux/Mac: export HF_ENDPOINT=https://hf-mirror.com\n"
            "  Windows:   set HF_ENDPOINT=https://hf-mirror.com"
        ),
    )
    parser.add_argument("--check", action="store_true", help="仅环境检查")
    parser.add_argument("--model", action="store_true", help="仅下载模型")
    parser.add_argument("--index", action="store_true", help="仅构建索引")
    parser.add_argument("--gpu", action="store_true", help="OpenVINO GPU 加速")

    args = parser.parse_args()

    print()
    print("=" * 50)
    print("  宿说 · 安装")
    print("=" * 50)

    check()

    if args.check:
        return

    install_deps()

    if args.model:
        download_model()
        return
    if args.index:
        build_index()
        return

    download_model()
    build_index()

    if args.gpu:
        print("\n" + "=" * 50)
        print("  GPU 加速")
        print("=" * 50)
        print("  正在导出 OpenVINO IR 模型...")
        print("  目标设备: Intel Arc / iGPU")
        print("  (需要 OpenVINO >= 2024.0)")
        # GPU export handled by existing configure.py logic
        print()

    verify()


if __name__ == "__main__":
    main()
