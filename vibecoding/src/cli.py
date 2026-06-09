"""
宿说 CLI — 多模态检索命令行工具.

用法:
    python -m vibecoding.src.cli search "游泳池干净吗"     # 文本 → 图片
    python -m vibecoding.src.cli search "游泳池" --granular  # 多粒度
    python -m vibecoding.src.cli search "游泳池" --rerank    # LTR 重排序
    python -m vibecoding.src.cli search "套房" --filter      # 分类器预过滤
    python -m vibecoding.src.cli reverse img_0               # 图片 → 文本
    python -m vibecoding.src.cli classify "房间很好"         # 房型分类
    python -m vibecoding.src.cli stats                       # 索引统计
    python -m vibecoding.src.cli demo                        # 启动 Web Demo
"""

import sys, os, json, argparse

_project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_project, "vibecoding", "src"))

from multimodal_api import MultimodalAPI


def cmd_search(args):
    api = MultimodalAPI()
    if args.rerank:
        r = api.prompt2image_rerank(args.query, topK=args.top)
        method = "CLIP + LTR"
    elif args.filter:
        r = api.prompt2image_filtered(args.query, topK=args.top)
        method = f"CLIP + Filter (pred: {r.get('predicted_room_type', '?')})"
    elif args.granular:
        r = api.prompt2image_granular(args.query, topK=args.top, granularity="room_type")
        method = "CLIP (room_type granular)"
    else:
        r = api.prompt2image(args.query, topK=args.top)
        method = "CLIP"

    print(f"\n{'='*60}")
    print(f"  Query: {args.query}")
    print(f"  Method: {method}")
    print(f"{'='*60}")
    for i, (img, s) in enumerate(zip(r["images"], r["score"])):
        print(f"  {i+1}. [{s:.4f}] {img}")
    print()


def cmd_reverse(args):
    api = MultimodalAPI()
    img_path = f"data/images/img_{args.image}.jpg"
    if not os.path.exists(os.path.join(_project, img_path)):
        print(f"Image not found: {img_path}")
        return

    r = api.image2text(img_path, topK=args.top)
    print(f"\n{'='*60}")
    print(f"  Image: {img_path}")
    print(f"{'='*60}")
    for i, (txt, s) in enumerate(zip(r["texts"], r["score"])):
        print(f"  {i+1}. [{s:.4f}] {txt[:100]}...")
    print()


def cmd_classify(args):
    api = MultimodalAPI()
    r = api.classify_room_type(args.text)
    print(f"\n  Text: {args.text[:80]}...")
    print(f"  Predicted: {r['room_type']} (conf={r['confidence']})")
    print(f"  All: {r['all_probs']}")


def cmd_stats(args):
    api = MultimodalAPI()
    s = api.stats
    print(f"\n  Images indexed: {s['images']}")
    print(f"  Texts indexed:  {s['texts']}")
    print(f"  Total comments: {s['comments_total']}")
    print()


def cmd_demo(args):
    import subprocess
    subprocess.run([sys.executable, os.path.join(_project, "demo.py")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="宿说 CLI")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("search", help="文本 → 图片")
    p.add_argument("query"); p.add_argument("--top", type=int, default=5)
    p.add_argument("--granular", action="store_true")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--filter", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("reverse", help="图片 → 文本")
    p.add_argument("image"); p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_reverse)

    p = sub.add_parser("classify", help="房型分类")
    p.add_argument("text")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("stats", help="索引统计")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("demo", help="启动 Web Demo")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
