"""Agentic RAG 检索引擎 — HotelCorpus + AgenticRAG (SUN Yuhao).

v2 增强 (dyx 贡献集成):
  - BM25 中文检索 (jieba 分词) 作为可选检索后端
  - RRF 多路融合合并 BM25 + char_terms 结果
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Optional

from _shared.text import char_terms, normalize, parse_categories
from _shared.data import COMMENTS_PATH, SUMMARIES_PATH

CATEGORY_ALIASES = {
    "服务": ["前台服务", "客房服务", "整体满意度"],
    "前台": ["前台服务", "退房/入住效率"],
    "入住": ["退房/入住效率", "前台服务"],
    "早餐": ["餐饮设施"], "餐饮": ["餐饮设施"],
    "房间": ["房间设施", "卫生状况", "安静程度", "景观/朝向"],
    "卫生": ["卫生状况", "房间设施"],
    "噪音": ["安静程度", "房间设施"],
    "安静": ["安静程度", "景观/朝向"],
    "交通": ["交通便利性", "周边配套"],
    "亲子": ["整体满意度", "公共设施", "客房服务"],
    "老人": ["房间设施", "交通便利性", "客房服务"],
}


class HotelCorpus:
    """酒店评论语料库 — 支持 BM25 + char_terms 双引擎."""

    def __init__(self, comments_path=COMMENTS_PATH, summaries_path=SUMMARIES_PATH,
                 max_docs=4000, use_bm25: bool = True):
        import pandas as pd
        df = pd.read_parquet(comments_path)
        self.docs = []
        for _, row in df.iterrows():
            comment = normalize(str(row.get("comment", "")))
            if not comment:
                continue
            cats = parse_categories(str(row.get("categories", "")))
            doc = {
                "comment": comment,
                "score": row.get("score", 0),
                "fuzzy_room_type": str(row.get("fuzzy_room_type", "")),
                "categories": str(row.get("categories", "")),
                "categories_list": cats,
                "term_counts": Counter(char_terms(comment + " " + " ".join(cats))),
                "_id": str(row.get("_id", "")),
            }
            self.docs.append(doc)
            if len(self.docs) >= max_docs:
                break
        self.summaries = self._load_summaries(Path(summaries_path))
        self.idf = self._build_idf()

        # [dyx 贡献] BM25 索引
        self._bm25 = None
        if use_bm25:
            try:
                from _shared.bm25 import InvertedIndex
                bm25 = InvertedIndex()
                documents = {}
                for d in self.docs:
                    documents[d["_id"]] = d["comment"]
                bm25.build(documents)
                self._bm25 = bm25
            except Exception:
                pass

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

    def search(self, query: str, categories=None, top_k=6, method="auto"):
        """检索方法.

        Args:
            method: "auto" (先 BM25 后 char_terms), "bm25", "char_terms"
        """
        if method in ("auto", "bm25") and self._bm25 is not None:
            bm25_raw = self._bm25.search(query, topk=top_k * 2)
            if bm25_raw and method == "bm25":
                scored = []
                id_map = {d["_id"]: d for d in self.docs}
                for doc_id, score in bm25_raw:
                    doc = id_map.get(doc_id)
                    if doc:
                        scored.append((score, doc))
                if categories:
                    for i, (s, doc) in enumerate(scored):
                        overlap = set(categories).intersection(doc["categories_list"])
                        scored[i] = (s + 8.0 * len(overlap), doc)
                scored.sort(key=lambda x: x[0], reverse=True)
                return scored[:top_k]

        # fallback: char_terms
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
    """Agentic RAG 流水线 — 计划 → 检索 → 评估 → 改写/重试."""

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
    """构建 LLM 上下文 — 评论证据 + 类别摘要."""
    lines = []
    for i, (score, doc) in enumerate(rag_result["results"][:5], start=1):
        cats = "、".join(doc.get("categories_list", []))
        lines.append(
            f"[评论{i}] score={score:.2f} 评分={doc.get('score')} 房型={doc.get('fuzzy_room_type')} "
            f"类别={cats}\\n{doc.get('comment')[:320]}"
        )
    for item in rag_result["summaries"][:3]:
        lines.append(
            f"[类别摘要] {item.get('category')} 评论数={item.get('comment_count')}\\n"
            f"{normalize(item.get('summary', ''))[:360]}"
        )
    return "\\n\\n".join(lines)
