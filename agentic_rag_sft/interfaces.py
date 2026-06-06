"""Interface adapters for the course Layer 1 and Layer 2 contracts.

This module keeps the implemented Agentic RAG/SFT work connectable to the
interfaces described in the project markdown files. The implementation is based
on exp3.ipynb outputs under the repository-level data directory.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from agentic_rag import COMMENTS_PATH, SUMMARIES_PATH, HotelCorpus, answer_question


def _doc_to_source(doc: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    return {
        "id": doc.get("_id") or doc.get("comment_id") or doc.get("id"),
        "content": doc.get("comment", ""),
        "score": doc.get("score"),
        "room_type": doc.get("fuzzy_room_type") or doc.get("room_type"),
        "publish_date": doc.get("publish_date") or doc.get("date"),
        "categories": doc.get("categories_list", []),
        "retrieval_score": round(score, 4) if score is not None else None,
    }


def retrieve(
    query: str,
    filters: dict | None = None,
    strategy: str = "hybrid",
    top_k: int = 10,
    comments_path: Path = COMMENTS_PATH,
    summaries_path: Path = SUMMARIES_PATH,
    max_docs: int = 4000,
) -> list[dict[str, Any]]:
    """Layer 1 retrieve() adapter.

    The current optimization reuses exp3.ipynb's filtered comments and category
    summaries, then adds category-guided lexical retrieval plus retry logic in
    ask(). The strategy argument is accepted for compatibility.
    """
    corpus = HotelCorpus(comments_path=comments_path, summaries_path=summaries_path, max_docs=max_docs)
    categories = corpus.infer_categories(query)
    results = corpus.search(query, categories=categories, top_k=top_k)

    rows = []
    for score, doc in results:
        source = _doc_to_source(doc, score)
        if filters:
            if filters.get("room_type") and filters["room_type"] not in str(source.get("room_type", "")):
                continue
        rows.append(source)
    return rows


def ask(
    query: str,
    filters: dict | None = None,
    strategy: str = "hybrid",
    top_k: int = 10,
    use_llm: bool = False,
    comments_path: Path = COMMENTS_PATH,
    summaries_path: Path = SUMMARIES_PATH,
    max_docs: int = 4000,
) -> dict[str, Any]:
    """Layer 1 ask() adapter matching the expected markdown contract."""
    start = time.time()
    output = answer_question(
        query=query,
        top_k=top_k,
        use_llm=use_llm,
        comments_path=comments_path,
        summaries_path=summaries_path,
        max_docs=max_docs,
    )
    result = output["result"]
    sources = [_doc_to_source(doc, score) for score, doc in result["results"][:top_k]]

    if filters:
        sources = [
            item
            for item in sources
            if not filters.get("room_type") or filters["room_type"] in str(item.get("room_type", ""))
        ]

    return {
        "answer": output["answer"],
        "sources": sources,
        "metrics": {
            "retrieval_count": len(result["results"]),
            "reranked_count": len(sources),
            "evidence_score": result["evidence_score"],
            "strategy": strategy,
            "agent_trace": result["trace"],
            "latency_breakdown": {"total": round(time.time() - start, 3)},
            "docs": output["docs"],
        },
    }


def embed(texts: list[str], text_type: str = "document") -> list[list[float]]:
    """Layer 1 embed() compatibility shim.

    exp3.ipynb's production vector retrieval used Chroma/DashVector assets. This
    lightweight repo adapter returns deterministic hash vectors so callers can
    keep the same interface during local smoke tests without external services.
    """
    vectors = []
    for text in texts:
        digest = hashlib.sha256((text_type + "\n" + text).encode("utf-8")).digest()
        vectors.append([round(byte / 255.0, 6) for byte in digest[:32]])
    return vectors


def advise(user_input: str, session_id: str = "default") -> dict[str, Any]:
    """Layer 2 advise() adapter.

    This wraps the Layer 1 Agentic RAG result as a scenario advisor response.
    A teammate's fuller Layer 2 implementation can call ask() directly and reuse
    the returned sources, metrics, and agent_trace fields.
    """
    layer1_result = ask(user_input, strategy="agentic_rag", top_k=6)
    trace = layer1_result["metrics"]["agent_trace"]
    categories = []
    if trace and isinstance(trace[0], dict):
        categories = trace[0].get("categories", [])

    dimensions = [
        {"name": category, "importance": "core", "query": f"{user_input} {category}"}
        for category in categories[:4]
    ]
    if not dimensions:
        dimensions = [{"name": "综合体验", "importance": "core", "query": user_input}]

    return {
        "scene": "酒店评论问答",
        "confidence": layer1_result["metrics"].get("evidence_score", 0.0),
        "dimensions": dimensions,
        "verdict": layer1_result["answer"].splitlines()[0] if layer1_result["answer"] else "",
        "reply": layer1_result["answer"],
        "sources": [
            {
                "dimension": dimensions[0]["name"],
                "content": source["content"],
                "ref_ids": [source["id"]],
            }
            for source in layer1_result["sources"]
        ],
        "metrics": {
            "session_id": session_id,
            "dimension_count": len(dimensions),
            "total_latency": layer1_result["metrics"]["latency_breakdown"]["total"],
            "layer1": layer1_result["metrics"],
        },
    }
