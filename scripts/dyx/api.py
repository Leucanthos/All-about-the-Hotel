"""
dyx API — 对外暴露的统一接口, 对齐 leuca.multimodal.api 和 yuhao.agentic.api 风格.

算法来源:
    lib.py, lib_syh.py → 重构提取至 engine.py + _shared/bm25.py + _shared/fusion.py

用法:
    from dyx.api import bm25_search, multi_path_search, expand_query

    # BM25 检索
    results = bm25_search(\"游泳池干净吗\", top_k=10)

    # 多路融合检索
    result = multi_path_search(\"早餐怎么样\", top_k=10)

    # 查询扩展
    queries = expand_query(\"适合带孩子吗\")
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from _shared.data import COMMENTS_PATH, PROJECT_ROOT
from _shared.bm25 import InvertedIndex
from _shared.fusion import rrf_fuse, hyde_generate
from _shared.llm import chat as llm_chat
from .engine import detect_intent, expand_query_llm, build_retrieval_pipeline


# ── 全局缓存 (避免重复构建 BM25 索引) ──

_index_cache = None


def _get_index(max_docs: int = 4000) -> InvertedIndex:
    """获取/构建 BM25 倒排索引 (缓存)."""
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    df = pd.read_parquet(COMMENTS_PATH)
    documents = {}
    for i, row in df.iterrows():
        comment = str(row.get('comment', '')).strip()
        if comment:
            documents[str(row.get('_id', i))] = comment
        if len(documents) >= max_docs:
            break

    index = InvertedIndex()
    index.build(documents)
    _index_cache = index
    return index


# ── 公开 API ──


def bm25_search(query: str, top_k: int = 10, max_docs: int = 4000) -> dict:
    """BM25 关键词检索."""
    start = time.time()
    index = _get_index(max_docs=max_docs)
    raw = index.search(query, topk=top_k)
    items = []
    for doc_id, score in raw:
        items.append({
            'doc_id': doc_id,
            'score': round(score, 4),
            'comment': index.documents.get(doc_id, '')[:300],
        })
    return {
        'query': query,
        'results': items,
        'method': 'BM25+jieba',
        'latency_ms': round((time.time() - start) * 1000),
        'total_docs': index.num_docs,
    }


def multi_path_search(
    query: str,
    top_k: int = 10,
    max_docs: int = 4000,
    use_vector: bool = True,
) -> dict:
    """多路融合检索: BM25 + 向量(可选) + RRF."""
    start = time.time()
    index = _get_index(max_docs=max_docs)
    intent = detect_intent(query)

    # BM25
    bm25_raw = index.search(query, topk=top_k * 2)
    bm25_items = [{
        'rank': i + 1, 'method': 'bm25',
        'doc_id': doc_id, 'score': round(s, 4),
        'comment': index.documents.get(doc_id, '')[:300],
    } for i, (doc_id, s) in enumerate(bm25_raw)]

    lists_to_fuse = [bm25_items]

    # 向量检索 (复用 leuca 的 CLIP 文本嵌入)
    vector_items = []
    if use_vector:
        try:
            from leuca.multimodal.engine import MultimodalRetriever
            retriever = MultimodalRetriever()
            raw_vec = retriever.search_images(query, top_k=top_k * 2)
            for i, item in enumerate(raw_vec):
                vector_items.append({
                    'rank': i + 1, 'method': 'vector',
                    'doc_id': item.get('image_id', f'img_{i}'),
                    'score': item.get('similarity', 0),
                    'comment': item.get('comment', '')[:300],
                })
            lists_to_fuse.append(vector_items)
        except Exception:
            pass

    # RRF 融合
    fused = rrf_fuse(lists_to_fuse, topk=top_k)

    return {
        'query': query,
        'intent': intent,
        'bm25_results': bm25_items[:top_k],
        'vector_results': vector_items[:top_k] if vector_items else [],
        'fused_results': fused,
        'latency_ms': round((time.time() - start) * 1000),
    }


def expand_query(query: str) -> list:
    """查询扩展 (LLM).\返回 [原始查询, 扩展1, ...]."""
    return expand_query_llm(query, chat_fn=llm_chat)


def intent_classify(query: str) -> dict:
    """细粒度意图分类."""
    return detect_intent(query)


# ── 快捷函数 (对齐 leuca/yuhao 模块级导出) ──

def search(query: str, top_k: int = 10) -> dict:
    """一站式检索: 多路融合."""
    return multi_path_search(query, top_k=top_k)
