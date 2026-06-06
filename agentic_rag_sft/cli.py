import argparse
import json
import time
from pathlib import Path

from agentic_rag import SUMMARIES_PATH, COMMENTS_PATH, answer_question
from build_sft_dataset import (
    DEFAULT_COMMENTS,
    DEFAULT_OUT_DIR,
    DEFAULT_REVERSE_QUERIES,
    DEFAULT_SUMMARIES,
    build_sft_dataset,
)


def add_q16_parser(subparsers):
    q16 = subparsers.add_parser("q16", help="方向16：构造酒店问答 SFT 数据集")
    q16_sub = q16.add_subparsers(dest="action", required=True)

    build = q16_sub.add_parser("build-sft", help="从 exp3.ipynb 数据产物构建 SFT train/val jsonl")
    build.add_argument("--reverse-queries", type=Path, default=DEFAULT_REVERSE_QUERIES)
    build.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    build.add_argument("--summaries", type=Path, default=DEFAULT_SUMMARIES)
    build.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    build.add_argument("--max-samples", type=int, default=1200)
    build.add_argument("--val-ratio", type=float, default=0.1)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--json", action="store_true", help="以 JSON 输出结果元数据")
    build.set_defaults(handler=handle_q16_build_sft)


def add_q18_parser(subparsers):
    q18 = subparsers.add_parser("q18", help="方向18：Agentic RAG 酒店评论问答")
    q18_sub = q18.add_subparsers(dest="action", required=True)

    ask = q18_sub.add_parser("ask", help="向酒店评论知识库提问")
    ask.add_argument("query", help="用户问题")
    ask.add_argument("--top-k", type=int, default=6)
    ask.add_argument("--comments", type=Path, default=COMMENTS_PATH)
    ask.add_argument("--summaries", type=Path, default=SUMMARIES_PATH)
    ask.add_argument("--max-docs", type=int, default=4000)
    ask.add_argument("--use-llm", action="store_true")
    ask.add_argument("--model", default="qwen-plus")
    ask.add_argument("--api-base", default=None)
    ask.add_argument("--show-trace", action="store_true")
    ask.add_argument("--json", action="store_true", help="以 JSON 输出答案、轨迹和元数据")
    ask.set_defaults(handler=handle_q18_ask)


def handle_q16_build_sft(args):
    result = build_sft_dataset(
        reverse_queries=args.reverse_queries,
        comments_path=args.comments,
        summaries_path=args.summaries,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.json:
        serializable = {key: str(value) if isinstance(value, Path) else value for key, value in result.items()}
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
        return

    print(f"方向16 SFT 数据构建完成：train={result['train']} val={result['val']} total={result['total']}")
    print(f"train: {result['train_path']}")
    print(f"val: {result['val_path']}")
    print(f"preview: {result['preview_path']}")


def handle_q18_ask(args):
    start = time.time()
    output = answer_question(
        args.query,
        top_k=args.top_k,
        use_llm=args.use_llm,
        model=args.model,
        api_base=args.api_base,
        comments_path=args.comments,
        summaries_path=args.summaries,
        max_docs=args.max_docs,
    )
    latency = time.time() - start
    result = output["result"]

    if args.json:
        payload = {
            "answer": output["answer"],
            "trace": result["trace"],
            "evidence_score": result["evidence_score"],
            "categories": result["categories"],
            "docs": output["docs"],
            "latency": round(latency, 2),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.show_trace:
        print("TRACE:")
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
        print()
    print(output["answer"])
    print(f"\nlatency={latency:.2f}s docs={output['docs']}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="CLI for exp3.ipynb hotel review outputs: q16 SFT data and q18 Agentic RAG."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)
    add_q16_parser(subparsers)
    add_q18_parser(subparsers)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
