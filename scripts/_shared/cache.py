"""_shared/cache.py — 全局共享资源缓存 (Singleton).

解决各自开发导致的重复加载问题:
  - parquet 数据只读一次
  - BM25 索引只构建一次
  - CLIP 模型只加载一次
  - 向量存储只打开一次

用法:
    from _shared.cache import cache
    df = cache.get_comments()       # 首次加载，之后复用
    bm25 = cache.get_bm25()         # 首次构建，之后复用
    retriever = cache.get_retriever()  # 首次加载 CLIP，之后复用
    store = cache.get_vector_store()
"""

import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from _shared.data import COMMENTS_PATH, SUMMARIES_PATH, VECTORS_DIR


class _ResourceCache:
    """线程安全的全局资源缓存."""

    def __init__(self):
        self._lock = threading.RLock()
        self._df: Optional[pd.DataFrame] = None
        self._bm25 = None
        self._retriever = None
        self._store = None
        self._comments = None

    # ── Parquet 数据 ──

    def get_comments(self, force_reload: bool = False) -> pd.DataFrame:
        """仅加载一次 parquet."""
        if self._df is not None and not force_reload:
            return self._df
        with self._lock:
            if self._df is not None and not force_reload:
                return self._df
            print("[cache] Loading comments parquet...")
            self._df = pd.read_parquet(COMMENTS_PATH)
            print(f"[cache] Loaded {len(self._df)} rows")
        return self._df

    # ── BM25 索引 (dyx 贡献) ──

    def get_bm25(self, force_rebuild: bool = False):
        """仅构建一次 BM25 倒排索引 (基于 dyx 的 InvertedIndex)."""
        if self._bm25 is not None and not force_rebuild:
            return self._bm25
        with self._lock:
            if self._bm25 is not None and not force_rebuild:
                return self._bm25
            from _shared.bm25 import InvertedIndex
            df = self.get_comments()
            documents = {}
            for i, row in df.iterrows():
                comment = str(row.get("comment", "")).strip()
                if comment:
                    documents[str(row.get("_id", i))] = comment
            print(f"[cache] Building BM25 index ({len(documents)} docs)...")
            idx = InvertedIndex()
            idx.build(documents)
            self._bm25 = idx
        return self._bm25

    # ── CLIP 多模态检索引擎 (Leuca 贡献) ──

    def get_retriever(self):
        """仅加载一次 Chinese-CLIP."""
        if self._retriever is not None:
            return self._retriever
        with self._lock:
            if self._retriever is not None:
                return self._retriever
            from leuca.multimodal.engine import MultimodalRetriever
            print("[cache] Loading CLIP retriever...")
            self._retriever = MultimodalRetriever()
        return self._retriever

    # ── NumPy 向量存储 (Leuca 贡献) ──

    def get_vector_store(self):
        from _shared.store import get_store
        if self._store is None:
            with self._lock:
                if self._store is None:
                    self._store = get_store()
        return self._store

    # ── 摘要数据 ──

    def get_summaries(self) -> dict:
        if self._comments is not None:
            return self._comments
        with self._lock:
            if self._comments is not None:
                return self._comments
            if SUMMARIES_PATH.exists():
                import json
                with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
                    self._comments = json.load(f)
            else:
                self._comments = {}
        return self._comments

    def reset(self):
        """清空缓存 (测试用)."""
        self._df = None
        self._bm25 = None
        self._retriever = None
        self._store = None
        self._comments = None


# 全局单例
cache = _ResourceCache()
