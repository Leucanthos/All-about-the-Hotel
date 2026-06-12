"""
宿说 — 酒店评论智能顾问 CLI.

用法:
    hotel search "游泳池干净吗"              # 关键词路由 (默认)
    hotel search "游泳池干净吗" --llm        # LLM 路由 (需 API key)
    hotel search "适合带老人吗" --rag        # 仅 Agentic RAG
    hotel search "套房 豪华装修" --mm        # 仅多模态检索

设置 LLM (推荐 DeepSeek):
    1. 获取 Key:  https://platform.deepseek.com/api_keys
    2. 充值:      https://platform.deepseek.com/top_up (¥1 起)
    3. 设置环境变量:  set DEEPSEEK_API_KEY=sk-xxx
    4. 使用:          hotel search "..." --llm

架构: CLI → _shared/router (Layer 2, LLM路由) → leuca/yuhao (Layer 1)
"""

import sys, os, json, argparse

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "scripts"))

from _shared.router import route, classify_intent


def cmd_search(args):
    result = route(
        args.query,
        top_k=args.top,
        strategies=None if args.strategy == "auto" else [args.strategy],
        use_llm=args.llm,
    )
    if not getattr(args, "json", True):
        if "_hint" in result:
            print(result.pop("_hint"))
        _pretty_print(result)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _pretty_print(r: dict):
    router_info = r.get("router", {})
    method = router_info.get("method", "keyword")
    label = f"LLM ({router_info.get('model', '?')})" if method == "llm" else "关键词"

    print(f"\n{'='*60}")
    print(f"  查询: {r['query']}")
    print(f"  路由: {label}")
    if method == "llm" and router_info.get("reason"):
        print(f"  理由: {router_info['reason']}")
    print(f"  意图: {r['intent']}  →  策略: {', '.join(r['strategies'])}")
    print(f"{'='*60}")

    for name, result in r.get("results", {}).items():
        if "error" in result:
            print(f"  [{name}] 错误: {result['error']}")
            continue
        if name == "multimodal":
            print(f"\n  [多模态 · {result.get('method', 'CLIP')}] ({result.get('latency_ms', 0)}ms)")
            for i, (img, s) in enumerate(zip(result["images"][:5], result["scores"][:5])):
                name_short = img.split("/")[-1] if "/" in img else img
                print(f"    {i+1}. [{s:.4f}] {name_short}")
        elif name == "rag":
            print(f"\n  [Agentic RAG] 证据分={result.get('evidence_score', 0)} ({result.get('latency_ms', 0)}ms)")
            print(f"  类别: {', '.join(result.get('categories', []))}")
            print(f"  {result.get('answer', '')[:500]}")

    meta = r.get("meta", {})
    print(f"\n  总延迟: {meta.get('total_latency_ms', 0)}ms")
    s = meta.get("index_stats", {})
    if s:
        print(f"  索引: {s.get('images', 0)} 图片, {s.get('texts', 0)} 文本")
    print()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hotel",
        description="宿说 — LLM 智能路由 · 多模态 + Agentic RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  hotel search \"游泳池\"\n  hotel search \"游泳池\" --llm\n"
               "LLM 配置:\n  1. 获取 Key: https://platform.deepseek.com/api_keys\n"
               "  2. 创建 .env: echo DEEPSEEK_API_KEY=sk-xxx > .env\n"
               "  3. 或设置环境变量: set DEEPSEEK_API_KEY=sk-xxx",
    )
    sub = parser.add_subparsers(dest="command")
    search = sub.add_parser("search", help="智能检索 (LLM 路由 or 关键词回退)")
    search.add_argument("query", help="自然语言查询")
    search.add_argument("--top", type=int, default=10, help="返回数量")
    search.add_argument("--llm", action="store_true",
                       help="启用 LLM (自动优先本地模型 > DeepSeek API)")
    search.add_argument("--strategy", choices=["auto", "multimodal", "rag", "all"],
                       default="auto", help="手动指定策略 (default: auto)")
    search.add_argument("--mm", action="store_const", const="multimodal", dest="strategy",
                       help="仅多模态 (Leuca)")
    search.add_argument("--rag", action="store_const", const="rag", dest="strategy",
                       help="仅 Agentic RAG (Yuhao)")
    search.add_argument("--no-json", action="store_false", dest="json",
                       help="人类可读输出")
    search.set_defaults(func=cmd_search, json=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
