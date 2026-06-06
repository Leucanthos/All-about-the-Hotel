import argparse
import ast
import csv
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
COMMENTS_PATH = DATA_ROOT / "filtered_comments.csv"
SUMMARIES_PATH = DATA_ROOT / "category_summaries.json"

CATEGORY_ALIASES = {
    "服务": ["前台服务", "客房服务", "整体满意度"],
    "前台": ["前台服务", "退房/入住效率"],
    "入住": ["退房/入住效率", "前台服务"],
    "早餐": ["餐饮设施"],
    "餐饮": ["餐饮设施"],
    "房间": ["房间设施", "卫生状况", "安静程度", "景观/朝向"],
    "卫生": ["卫生状况", "房间设施"],
    "噪音": ["安静程度", "房间设施"],
    "安静": ["安静程度", "景观/朝向"],
    "交通": ["交通便利性", "周边配套"],
    "亲子": ["整体满意度", "公共设施", "客房服务"],
    "老人": ["房间设施", "交通便利性", "客房服务"],
}


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def normalize(text: str) -> str:
    return " ".join((text or "").split())


def parse_categories(value: str):
    try:
        return [str(x) for x in ast.literal_eval(value or "[]")]
    except Exception:
        return []


def char_terms(text: str):
    text = normalize(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = ["".join(chinese[i : i + 2]) for i in range(max(0, len(chinese) - 1))]
    return words + chinese + bigrams


class HotelCorpus:
    def __init__(self, comments_path=COMMENTS_PATH, summaries_path=SUMMARIES_PATH, max_docs=4000):
        self.docs = []
        for row in read_csv(Path(comments_path)):
            comment = normalize(row.get("comment", ""))
            if not comment:
                continue
            row["comment"] = comment
            row["categories_list"] = parse_categories(row.get("categories", ""))
            row["term_counts"] = Counter(char_terms(comment + " " + " ".join(row["categories_list"])))
            self.docs.append(row)
            if len(self.docs) >= max_docs:
                break
        self.summaries = self._load_summaries(Path(summaries_path))
        self.idf = self._build_idf()

    def _load_summaries(self, path: Path):
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        return {r.get("category", ""): r for r in rows if r.get("category")}

    def _build_idf(self):
        df = Counter()
        for doc in self.docs:
            df.update(doc["term_counts"].keys())
        n = max(1, len(self.docs))
        return {term: math.log((n + 1) / (freq + 0.5)) for term, freq in df.items()}

    def infer_categories(self, query: str):
        cats = []
        for key, values in CATEGORY_ALIASES.items():
            if key in query:
                cats.extend(values)
        return list(dict.fromkeys(cats))

    def search(self, query: str, categories=None, top_k=6):
        q_terms = Counter(char_terms(query))
        category_set = set(categories or [])
        scored = []
        for doc in self.docs:
            score = 0.0
            for term, qtf in q_terms.items():
                score += qtf * doc["term_counts"].get(term, 0) * self.idf.get(term, 0.1)
            if category_set:
                overlap = category_set.intersection(doc["categories_list"])
                score += 8.0 * len(overlap)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def summaries_for(self, categories):
        rows = []
        for category in categories:
            item = self.summaries.get(category)
            if item:
                rows.append(item)
        return rows


class AgenticRAG:
    def __init__(self, corpus: HotelCorpus):
        self.corpus = corpus

    def plan(self, query: str):
        categories = self.corpus.infer_categories(query)
        strategy = "category_guided" if categories else "broad_lexical"
        return {"strategy": strategy, "categories": categories, "query": query}

    def evaluate(self, query: str, results):
        if not results:
            return {"score": 0.0, "need_retry": True, "reason": "no evidence"}
        top_score = results[0][0]
        comments = " ".join(doc["comment"] for _, doc in results[:3])
        coverage = len(set(char_terms(query)).intersection(char_terms(comments)))
        score = min(1.0, top_score / 45.0 + coverage / 40.0)
        return {
            "score": round(score, 3),
            "need_retry": score < 0.45,
            "reason": "low evidence coverage" if score < 0.45 else "enough evidence",
        }

    def rewrite(self, query: str, categories):
        hints = " ".join(categories) if categories else "服务 房间 交通 早餐 卫生 噪音"
        return f"{query} {hints} 优点 缺点 建议"

    def run(self, query: str, top_k=6):
        trace = []
        plan = self.plan(query)
        trace.append({"step": "plan", **plan})
        results = self.corpus.search(plan["query"], plan["categories"], top_k=top_k)
        eval_result = self.evaluate(query, results)
        trace.append({"step": "evaluate", **eval_result})

        if eval_result["need_retry"]:
            retry_query = self.rewrite(query, plan["categories"])
            retry_categories = plan["categories"] or self.corpus.infer_categories(retry_query)
            trace.append({"step": "rewrite", "query": retry_query, "categories": retry_categories})
            retry_results = self.corpus.search(retry_query, retry_categories, top_k=top_k)
            retry_eval = self.evaluate(query, retry_results)
            trace.append({"step": "retry_evaluate", **retry_eval})
            if retry_eval["score"] >= eval_result["score"]:
                results = retry_results
                eval_result = retry_eval

        categories = plan["categories"]
        if not categories and results:
            for _, doc in results[:3]:
                categories.extend(doc["categories_list"])
            categories = list(dict.fromkeys(categories))[:4]

        return {
            "query": query,
            "trace": trace,
            "evidence_score": eval_result["score"],
            "categories": categories,
            "summaries": self.corpus.summaries_for(categories[:4]),
            "results": results,
        }


def build_context(rag_result):
    lines = []
    for i, (score, doc) in enumerate(rag_result["results"][:5], start=1):
        cats = "、".join(doc.get("categories_list", []))
        lines.append(
            f"[评论{i}] score={score:.2f} 评分={doc.get('score')} 房型={doc.get('fuzzy_room_type')} "
            f"类别={cats}\n{doc.get('comment')[:320]}"
        )
    for item in rag_result["summaries"][:3]:
        lines.append(
            f"[类别摘要] {item.get('category')} 评论数={item.get('comment_count')}\n"
            f"{normalize(item.get('summary', ''))[:360]}"
        )
    return "\n\n".join(lines)


def fallback_answer(rag_result):
    context = build_context(rag_result)
    cats = "、".join(rag_result["categories"][:4]) or "综合体验"
    risks = []
    positives = []
    negative_words = [
        "老旧",
        "噪音",
        "排队",
        "卫生",
        "不便",
        "异味",
        "施工",
        "慢",
        "无奈",
        "坏",
        "吐槽",
        "差",
        "投诉",
        "不专业",
        "混乱",
        "失望",
    ]
    for _, doc in rag_result["results"][:5]:
        text = doc["comment"]
        try:
            score_value = float(doc.get("score") or 0)
        except ValueError:
            score_value = 0
        has_negative = any(k in text for k in negative_words)
        if has_negative or score_value < 4.0:
            risks.append(text[:90])
        elif score_value >= 4.5:
            positives.append(text[:90])
    if not positives:
        positives = [
            normalize(item.get("summary", ""))[:90]
            for item in rag_result["summaries"][:2]
            if item.get("summary")
        ]
    positives = positives[:2] or [doc["comment"][:90] for _, doc in rag_result["results"][:2]]
    risks = risks[:2] or ["当前命中的评论中负面证据较少，仍建议结合具体房型和入住日期确认。"]
    return (
        f"结论：这个问题主要涉及{cats}。证据覆盖分为 {rag_result['evidence_score']}，可以作为回答依据。\n\n"
        f"主要亮点：\n- " + "\n- ".join(positives) + "\n\n"
        f"需要提示的风险：\n- " + "\n- ".join(risks) + "\n\n"
        f"建议：预订时备注具体需求，例如安静房、翻新房型、老人小孩出行便利；入住高峰期提前到店或预留办理时间。\n\n"
        f"参考证据节选：\n{context[:900]}"
    )


def call_openai_compatible(prompt, model, api_key, api_base):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严谨的酒店评论问答助手，只能依据证据回答。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def answer_with_optional_llm(rag_result, use_llm=False, model="qwen-plus", api_base=None):
    if not use_llm:
        return fallback_answer(rag_result)
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_answer(rag_result)
    api_base = api_base or os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    prompt = (
        f"用户问题：{rag_result['query']}\n\n"
        f"Agent 检索轨迹：{json.dumps(rag_result['trace'], ensure_ascii=False)}\n\n"
        f"证据：\n{build_context(rag_result)}\n\n"
        "请输出：结论、证据依据、风险提醒、行动建议。"
    )
    try:
        return call_openai_compatible(prompt, model, api_key, api_base)
    except Exception as exc:
        return fallback_answer(rag_result) + f"\n\n[LLM 调用失败，已回退到本地模板：{exc}]"


def answer_question(
    query: str,
    top_k: int = 6,
    use_llm: bool = False,
    model: str = "qwen-plus",
    api_base: str | None = None,
    comments_path: Path = COMMENTS_PATH,
    summaries_path: Path = SUMMARIES_PATH,
    max_docs: int = 4000,
):
    """Run the Agentic RAG pipeline and return both answer text and metadata."""
    corpus = HotelCorpus(comments_path=comments_path, summaries_path=summaries_path, max_docs=max_docs)
    agent = AgenticRAG(corpus)
    result = agent.run(query, top_k=top_k)
    answer = answer_with_optional_llm(result, use_llm, model, api_base)
    return {
        "answer": answer,
        "result": result,
        "docs": len(corpus.docs),
    }


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG demo for hotel review QA.")
    parser.add_argument("query", nargs="?", default="这家酒店适合带老人小孩入住吗？房间和交通怎么样？")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    start = time.time()
    output = answer_question(
        args.query,
        top_k=args.top_k,
        use_llm=args.use_llm,
        model=args.model,
        api_base=args.api_base,
    )
    result = output["result"]

    if args.show_trace:
        print("TRACE:")
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
        print()
    print(output["answer"])
    print(f"\nlatency={time.time() - start:.2f}s docs={output['docs']}")


if __name__ == "__main__":
    main()
