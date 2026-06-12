"""宿说 — 酒店评论智能顾问 CLI (v2: +dyx BM25/多路融合).

用法:
    hotel search "游泳池干净吗"              # 关键词路由 (默认, dyx 意图分类)
    hotel search "游泳池干净吗" --llm        # LLM 路由 (需 API key)
    hotel search "适合带老人吗" --rag        # 仅 Agentic RAG (Yuhao)
    hotel search "套房 豪华装修" --mm        # 仅多模态 (Leuca)
    hotel search "停车方便吗" --bm25         # 仅 BM25 关键词 (dyx)
    hotel search "早餐怎么样" --fusion       # BM25+CLIP 多路融合 (dyx)
    hotel search "隔音好吗" --strategy all   # 全部策略
"""

import sys, os, json, argparse

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "scripts"))

from _shared.router import route

# 策略名称 → 中文标签
_STRATEGY_LABELS = {
    "multimodal": "多模态 CLIP",
    "rag":        "Agentic RAG",
    "bm25":       "BM25 关键词",
    "fusion":     "多路融合 (BM25+CLIP)",
}

_STRATEGY_CHOICES = ["auto", "multimodal", "rag", "bm25", "fusion", "all"]


def cmd_search(args):
    strategies = None if args.strategy == "auto" else [args.strategy]
    result = route(args.query, top_k=args.top, strategies=strategies, use_llm=args.llm)

    if not getattr(args, "json", True):
        if "_hint" in result:
            print(result.pop("_hint"))
        _pretty_print(result)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _pretty_print(r: dict):
    """人类可读的格式化输出."""
    router_info = r.get("router", {})
    method = router_info.get("method", "keyword")
    label = f"LLM ({router_info.get('model', '?')})" if method == "llm" else "关键词"

    print(f"\n{'='*60}")
    print(f"  查询: {r['query']}")
    print(f"  路由: {label}")
    if method == "llm" and router_info.get("reason"):
        print(f"  理由: {router_info['reason']}")
    # 显示 dyx 细粒度意图
    intent_detail = router_info.get("intent_detail", {})
    if intent_detail:
        cats = "、".join(intent_detail.get("categories", []))
        print(f"  意图: {intent_detail.get('primary', '?')}  [{cats}]  "
              f"置信度: {intent_detail.get('confidence', 0)*100:.0f}%")
    else:
        print(f"  意图: {r['intent']}")
    print(f"  策略: {', '.join(r.get('strategies', []))}")
    print(f"{'='*60}")

    for name, result in r.get("results", {}).items():
        label = _STRATEGY_LABELS.get(name, name)
        latency = result.get("latency_ms", 0)

        if "error" in result:
            print(f"\n  [{label}] 错误: {result['error']}")
            continue

        # 多模态结果
        if name == "multimodal":
            method_name = result.get("method", "CLIP")
            print(f"\n  [{label}] ({method_name}) [{latency}ms]")
            for i, (img, s) in enumerate(zip(result.get("images", [])[:5],
                                              result.get("scores", [])[:5])):
                short = img.split("/")[-1] if "/" in img else img
                print(f"    {i+1}. [{s:.4f}] {short}")

        # RAG 结果
        elif name == "rag":
            print(f"\n  [{label}] 证据分={result.get('evidence_score', 0)} [{latency}ms]")
            print(f"  类别: {', '.join(result.get('categories', []))}")
            answer = result.get("answer", "")
            if answer:
                print(f"  回答: {answer[:500]}")

        # BM25 结果 (dyx)
        elif name == "bm25":
            items = result.get("results", [])
            total = result.get("total_docs", 0)
            print(f"\n  [{label}] {len(items)} 条结果 (索引共 {total} 篇) [{latency}ms]")
            for i, item in enumerate(items[:8]):
                comment = item.get("comment", "")[:120]
                print(f"    {i+1}. [{item.get('score', 0):.4f}] {comment}...")

        # 多路融合结果 (dyx)
        elif name == "fusion":
            fused = result.get("fused_results", [])
            bm25_n = result.get("bm25_count", 0)
            clip_n = result.get("clip_count", 0)
            print(f"\n  [{label}] {len(fused)} 条 (BM25 {bm25_n} 条 + CLIP {clip_n} 条) [{latency}ms]")
            for i, item in enumerate(fused[:8]):
                # 包含来自 RRF 融合的 comment
                comment = item.get("comment", "")[:120]
                fid = item.get("doc_id", "?")
                fs = item.get("fused_score", 0)
                print(f"    {i+1}. [{fs:.4f}] [{fid}] {comment}...")

    # 元信息
    meta = r.get("meta", {})
    print(f"\n  总延迟: {meta.get('total_latency_ms', 0)}ms")
    s = meta.get("index_stats", {})
    if s:
        print(f"  索引: {s.get('images', 0)} 图片, {s.get('texts', 0)} 文本")
    engines = meta.get("engines", {})
    if engines:
        contribs = set()
        for e_name, e_info in engines.items():
            if "contributor" in e_info:
                contribs.add(e_info["contributor"])
        if contribs:
            print(f"  算法贡献: {', '.join(sorted(contribs))}")
    print()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hotel",
        description="宿说 — LLM 智能路由 · 多模态 + Agentic RAG + BM25 增强",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               '  hotel search "游泳池"\n'
               '  hotel search "游泳池" --llm\n'
               '  hotel search "早餐" --bm25\n'
               '  hotel search "房间" --fusion\n'
               "LLM 配置:\n"
               "  1. 获取 Key: https://platform.deepseek.com/api_keys\n"
               "  2. 创建 .env: echo DEEPSEEK_API_KEY=sk-xxx > .env\n"
               "  3. 或设置环境变量: set DEEPSEEK_API_KEY=sk-xxx",
    )
    sub = parser.add_subparsers(dest="command")
    search = sub.add_parser("search", help="智能检索 (关键词/LLM路由)")
    search.add_argument("query", help="自然语言查询")
    search.add_argument("--top", type=int, default=10, help="返回数量")
    search.add_argument("--llm", action="store_true",
                        help="启用 LLM (自动优先本地模型 > DeepSeek API)")
    search.add_argument("--strategy", choices=_STRATEGY_CHOICES,
                        default="auto",
                        help="检索策略 (default: auto, 使用 dyx 意图分类自动选择)")
    # 快捷别名
    search.add_argument("--mm", action="store_const", const="multimodal",
                        dest="strategy", help="仅多模态 (Leuca)")
    search.add_argument("--rag", action="store_const", const="rag",
                        dest="strategy", help="仅 Agentic RAG (Yuhao)")
    search.add_argument("--bm25", action="store_const", const="bm25",
                        dest="strategy", help="仅 BM25 关键词 (dyx)")
    search.add_argument("--fusion", action="store_const", const="fusion",
                        dest="strategy", help="BM25+CLIP 多路融合 (dyx)")
    search.add_argument("--all", action="store_const", const="all",
                        dest="strategy", help="全部策略")
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

