"""_shared/retriever.py — 统一检索编排器 (v3: +dyx 聚类提升).

解决各自开发导致的调用分散问题:
  - 一次查询, 智能调度所有引擎
  - 共享缓存避免重复加载
  - RRF 融合在编排层统一完成
  - 每位成员的贡献保持独立可调
  - 聚类感知 (dyx): 用预计算主题集群提升相关文档得分

用法:
    from _shared.retriever import retrieve
    result = retrieve("游泳池干净吗")                          # auto
    result = retrieve("早餐好吃吗", strategies=["rag"])
    result = retrieve("房间照片", strategies=["multimodal", "bm25"])
    result = retrieve("服务怎么样", use_clusters=True)         # 启用聚类提升
"""

import time
from typing import List, Optional

from _shared.cache import cache
from _shared.fusion import rrf_fuse


def retrieve(
    query: str,
    top_k: int = 10,
    strategies: Optional[List[str]] = None,
    use_llm: bool = False,
    use_clusters: bool = False,
) -> dict:
    """统一检索入口.

    Args:
        query: 自然语言查询
        top_k: 每路返回数量
        strategies: 检索策略列表, None=auto 检测
        use_llm: 是否启用 LLM 生成回答
        use_clusters: 是否启用聚类感知提升 (dyx 贡献)

    Returns:
        {query, intent, strategies, results, engines, meta}
    """
    start = time.time()

    # ── 意图检测 (dyx 贡献) ──
    from _shared.router import classify_intent_detailed
    intent_info = classify_intent_detailed(query) if strategies is None else None

    if strategies is None:
        if intent_info:
            strategies = intent_info["strategies"]
        else:
            strategies = ["multimodal", "rag"]
    elif "all" in strategies:
        strategies = ["multimodal", "rag", "bm25"]

    strategies = list(dict.fromkeys(strategies))

    # ── 可选: 聚类提升 (dyx) ──
    cluster_boost = None
    cluster_hints = []
    if use_clusters:
        try:
            from _shared.clusters import get_cluster_boost
            cb = get_cluster_boost()
            if cb.n_clusters > 0:
                cluster_boost = cb
                cluster_hints = cb.match_query(query)
        except Exception:
            pass

    # ── 并行执行各路检索 ──
    results = {}
    engine_info = {}

    for name in strategies:
        t0 = time.time()
        if name == "multimodal":
            r, info = _run_multimodal(query, top_k)
        elif name == "rag":
            r, info = _run_rag(query, min(top_k, 8), use_llm, cluster_boost)
        elif name == "bm25":
            r, info = _run_bm25(query, top_k, cluster_boost)
        elif name == "fusion":
            r, info = _run_fusion(query, top_k)
        else:
            continue
        if r:
            r["latency_ms"] = round((time.time() - t0) * 1000)
        results[name] = r
        engine_info[name] = info

    # ── 元信息 ──
    meta = {
        "total_latency_ms": round((time.time() - start) * 1000),
        "strategies_run": strategies,
    }
    if cluster_hints:
        meta["cluster_hints"] = [c for c in cluster_hints[:5]]
    try:
        s = cache.get_retriever().stats if strategies and "multimodal" in strategies else {}
        meta["index_stats"] = s
    except Exception:
        pass

    return {
        "query": query,
        "intent": intent_info or {},
        "strategies": strategies,
        "results": results,
        "engines": engine_info,
        "meta": meta,
    }


# ── 各引擎调用 ──

def _run_multimodal(query: str, top_k: int = 10):
    """多模态检索 — Leuca 贡献."""
    try:
        retriever = cache.get_retriever()
        raw = retriever.search_images(query, top_k=top_k)
        return {
            "images": [r.get("comment", "") for r in raw],
            "image_ids": [r.get("image_id", "") for r in raw],
            "scores": [r.get("similarity", 0) for r in raw],
            "method": "Chinese-CLIP",
            "total": len(raw),
        }, {"contributor": "Leuca"}
    except Exception as e:
        return {"error": str(e)}, {"contributor": "Leuca", "error": str(e)}


def _run_rag(query: str, top_k: int = 8, use_llm: bool = False,
             cluster_boost=None):
    """BM25 先试 → 不够再升级 RAG (Yuhao 贡献 + 编排优化).

    策略:
      1. 先跑 BM25 (轻量, ~1ms)
      2. 如果 BM25 结果充分 (top score >= 4.0) → 直接返回, 跳过 RAG
      3. 如果 BM25 结果不足 → 升级到完整 RAG (含 Agent 循环)
    """
    BM25_CONFIDENCE_THRESHOLD = 4.0
    try:
        index = cache.get_bm25()
        bm25_raw = index.search(query, topk=top_k)
        bm25_top_score = bm25_raw[0][1] if bm25_raw else 0.0
        bm25_has_results = len(bm25_raw) >= 3
        bm25_high_confidence = bm25_top_score >= BM25_CONFIDENCE_THRESHOLD

        if bm25_has_results and bm25_high_confidence:
            df = cache.get_comments()
            items = []
            for doc_id, score in bm25_raw[:top_k]:
                # 聚类提升 (dyx)
                if cluster_boost is not None:
                    score = cluster_boost.boost_score(query, doc_id, score)

                row = df[df["_id"].astype(str) == doc_id]
                comment = str(row["comment"].values[0]) if not row.empty else ""
                items.append({"doc_id": doc_id, "score": round(score, 4), "comment": comment[:300]})

            # 按提升后分数重排序
            items.sort(key=lambda x: x["score"], reverse=True)

            categories = set()
            for _, doc_id in bm25_raw[:5]:
                row = df[df["_id"].astype(str) == doc_id]
                if not row.empty and "categories" in row.columns:
                    import ast
                    try:
                        cats = ast.literal_eval(str(row["categories"].values[0]))
                        if isinstance(cats, list):
                            categories.update(cats)
                    except Exception:
                        pass

            return {
                "answer": f"从 {len(items)} 条评论中找到相关证据.",
                "evidence_score": round(min(bm25_top_score / 10.0, 1.0), 3),
                "categories": list(categories)[:4],
                "trace": [{"step": "bm25_first", "top_score": bm25_top_score,
                          "escalated": False}],
                "sources": [{"score": item["score"], "comment": item["comment"]}
                           for item in items[:5]],
                "_bm25_escalated": False,
            }, {"contributor": "dyx+Yuhao",
                "note": "BM25 sufficient, RAG skipped"}
    except Exception:
        pass

    # Step 2: BM25 不足, 升级 RAG (Yuhao)
    try:
        from yuhao.agentic.api import answer_question
        output = answer_question(query, top_k=top_k, use_llm=use_llm)
        result = output["result"]
        return {
            "answer": output.get("answer", "")[:2000],
            "evidence_score": result.get("evidence_score", 0),
            "categories": result.get("categories", []),
            "trace": result.get("trace", []),
            "sources": [{"score": round(s, 2), "comment": doc.get("comment", "")[:200]}
                       for s, doc in result.get("results", [])[:5]],
            "_bm25_escalated": True,
        }, {"contributor": "Yuhao", "docs": output.get("docs", 0)}
    except Exception as e:
        return {"error": str(e)}, {"contributor": "Yuhao", "error": str(e)}


def _run_bm25(query: str, top_k: int = 10, cluster_boost=None):
    """BM25 关键词检索 — dyx 贡献 (可选聚类提升)."""
    try:
        index = cache.get_bm25()
        raw = index.search(query, topk=top_k)
        df = cache.get_comments()
        items = []
        for doc_id, score in raw:
            if cluster_boost is not None:
                score = cluster_boost.boost_score(query, doc_id, score)
            row = df[df["_id"].astype(str) == doc_id]
            comment = str(row["comment"].values[0]) if not row.empty else ""
            items.append({"doc_id": doc_id, "score": round(score, 4), "comment": comment[:300]})

        items.sort(key=lambda x: x["score"], reverse=True)

        return {"results": items, "total": len(items), "method": "BM25+jieba"}, {
            "contributor": "dyx", "index_docs": index.num_docs}
    except Exception as e:
        return {"error": str(e)}, {"contributor": "dyx", "error": str(e)}


def _run_fusion(query: str, top_k: int = 10):
    """多路融合 (BM25 + CLIP) — dyx 贡献."""
    try:
        index = cache.get_bm25()
        bm25_raw = index.search(query, topk=top_k * 2)
        bm25_items = [{
            "rank": i + 1, "method": "bm25",
            "doc_id": doc_id, "score": round(s, 4),
        } for i, (doc_id, s) in enumerate(bm25_raw)]

        retriever = cache.get_retriever()
        clip_raw = retriever.search_images(query, top_k=top_k * 2)
        clip_items = [{
            "rank": i + 1, "method": "clip",
            "doc_id": r.get("image_id", f"img_{i}"),
            "score": r.get("similarity", 0),
        } for i, r in enumerate(clip_raw)]

        fused = rrf_fuse([bm25_items, clip_items], topk=top_k)
        return {
            "fused_results": fused,
            "bm25_count": len(bm25_items),
            "clip_count": len(clip_items),
            "method": "RRF fusion",
        }, {"contributor": "dyx"}
    except Exception as e:
        return {"error": str(e)}, {"contributor": "dyx", "error": str(e)}
