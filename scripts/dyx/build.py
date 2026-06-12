"""
dyx 索引构建工具 — BM25 倒排索引 + 构建/序列化.

对齐 leuca.multimodal.build 和 yuhao.sft.build 风格.

用法:
    from dyx.build import build_bm25_index
    index = build_bm25_index(max_docs=4000)
    index.save("data/models/bm25_index.pkl")
"""

import pickle
from pathlib import Path

import pandas as pd

from _shared.data import COMMENTS_PATH
from _shared.bm25 import InvertedIndex


def build_bm25_index(
    comments_path: Path = COMMENTS_PATH,
    max_docs: int = 4000,
    save_path: Path = None,
) -> InvertedIndex:
    """从 parquet 构建 BM25 倒排索引."""
    df = pd.read_parquet(comments_path)
    documents = {}
    for i, row in df.iterrows():
        comment = str(row.get('comment', '')).strip()
        if comment:
            documents[str(row.get('_id', i))] = comment
        if len(documents) >= max_docs:
            break

    index = InvertedIndex()
    index.build(documents)
    print(f"[dyx.build] BM25 index built: {index.num_docs} docs, {len(index.index)} terms")

    if save_path:
        index.save(str(save_path))
        print(f"[dyx.build] Saved to {save_path}")

    return index


def load_bm25_index(path: str) -> InvertedIndex:
    """加载序列化的 BM25 索引."""
    index = InvertedIndex()
    index.load(path)
    print(f"[dyx.build] Loaded: {index.num_docs} docs, {len(index.index)} terms")
    return index
