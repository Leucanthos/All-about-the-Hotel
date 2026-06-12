import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from _shared.text import parse_categories, read_csv, short_text
from _shared.data import COMMENTS_PATH, SUMMARIES_PATH, REVERSE_QUERIES_PATH, SFT_DIR

DEFAULT_REVERSE_QUERIES = REVERSE_QUERIES_PATH
DEFAULT_COMMENTS = COMMENTS_PATH
DEFAULT_SUMMARIES = SUMMARIES_PATH
DEFAULT_OUT_DIR = SFT_DIR

SYSTEM_PROMPT = (
    "你是广州花园酒店评论问答助手。回答必须基于给定评论证据，"
    "先给结论，再分点说明优点、风险和适用建议；不要编造评论中没有的信息。"
)


def load_comments(path: Path):
    import pandas as pd
    df = pd.read_parquet(path)
    comments = {}
    by_category = defaultdict(list)
    for _, row in df.iterrows():
        cid = str(row.get("_id", ""))
        if not cid:
            continue
        doc = {"_id": cid, "comment": short_text(str(row.get("comment", ""))),
               "score": str(row.get("score", "")),
               "fuzzy_room_type": str(row.get("fuzzy_room_type", "")),
               "room_type": str(row.get("room_type", "")),
               "travel_type": str(row.get("travel_type", "")),
               "categories": str(row.get("categories", ""))}
        comments[cid] = doc
        for category in parse_categories(doc["categories"]):
            by_category[category].append(doc)
    return comments, by_category


def load_summaries(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r.get("category", ""): r for r in rows if r.get("category")}


def build_answer(query_row, comment_row, summaries):
    categories = parse_categories(comment_row.get("categories", ""))
    score = comment_row.get("score", "")
    room = comment_row.get("fuzzy_room_type") or comment_row.get("room_type") or "未知房型"
    travel = comment_row.get("travel_type") or "未知出行类型"
    comment = short_text(query_row.get("comment") or comment_row.get("comment", ""))

    summary_lines = []
    for category in categories[:2]:
        item = summaries.get(category)
        if item:
            summary_lines.append(f"- {category}: {short_text(item.get('summary', ''), 180)}")

    category_text = "、".join(categories) if categories else "综合体验"
    summary_text = "\n".join(summary_lines) if summary_lines else "- 暂无类别摘要，仅依据原始评论回答。"

    return (
        f"结论：这个问题主要涉及{category_text}。从现有评论看，需要同时说明亮点和潜在风险。\n\n"
        f"证据：\n"
        f"- 原始评论：{comment}\n"
        f"- 评分/房型/出行：{score} 分，{room}，{travel}\n"
        f"- 类别摘要：\n{summary_text}\n\n"
        f"回答建议：\n"
        f"1. 如果用户关心体验优势，应优先说明评论中明确出现的服务、设施或位置优点。\n"
        f"2. 如果评论包含设施老旧、噪音、排队、卫生等负面信息，需要直接提示风险。\n"
        f"3. 最后给出可执行建议，例如提前备注安静房、确认房型、避开高峰办理入住。"
    )


def make_conversation(query, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ],
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": query},
            {"from": "gpt", "value": answer},
        ],
    }


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sft_dataset(
    reverse_queries: Path = DEFAULT_REVERSE_QUERIES,
    comments_path: Path = DEFAULT_COMMENTS,
    summaries_path: Path = DEFAULT_SUMMARIES,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_samples: int = 1200,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Build train/val SFT jsonl files and return output metadata."""
    comments, _ = load_comments(Path(comments_path))
    summaries = load_summaries(Path(summaries_path))

    rows = []
    seen = set()
    for query_row in read_csv(Path(reverse_queries)):
        query = short_text(query_row.get("query", ""), 260)
        cid = query_row.get("comment_id")
        if not query or not cid or cid not in comments:
            continue
        key = (query, cid)
        if key in seen:
            continue
        seen.add(key)
        answer = build_answer(query_row, comments[cid], summaries)
        rows.append(make_conversation(query, answer))
        if len(rows) >= max_samples:
            break

    random.Random(seed).shuffle(rows)
    val_size = max(1, int(len(rows) * val_ratio)) if len(rows) > 10 else 0
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sft_train.jsonl"
    val_path = out_dir / "sft_val.jsonl"
    preview_path = out_dir / "sft_preview.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    with preview_path.open("w", encoding="utf-8") as f:
        json.dump(rows[:5], f, ensure_ascii=False, indent=2)

    return {
        "train": len(train_rows),
        "val": len(val_rows),
        "total": len(rows),
        "train_path": train_path,
        "val_path": val_path,
        "preview_path": preview_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Build hotel review SFT data from Exp3 notebook outputs.")
    parser.add_argument("--reverse-queries", type=Path, default=DEFAULT_REVERSE_QUERIES)
    parser.add_argument("--comments", type=Path, default=DEFAULT_COMMENTS)
    parser.add_argument("--summaries", type=Path, default=DEFAULT_SUMMARIES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-samples", type=int, default=1200)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_sft_dataset(
        reverse_queries=args.reverse_queries,
        comments_path=args.comments,
        summaries_path=args.summaries,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"train={result['train']} val={result['val']}")
    print(f"wrote {result['train_path']}")
    print(f"wrote {result['val_path']}")


if __name__ == "__main__":
    main()
