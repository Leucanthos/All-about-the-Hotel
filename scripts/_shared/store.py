"""
NumPy 向量存储 — 替代 ChromaDB，解决 Windows HNSW 持久化 bug.

特性:
  - 纯 NumPy 余弦相似度检索
  - 无外部依赖问题，跨平台稳定
  - 支持任意规模的向量
  - 批量查询 (一次矩阵乘法)
"""

import os, json, numpy as np
from typing import List, Dict, Optional


class NumpyVectorStore:
    """轻量级向量存储 + 余弦检索."""

    def __init__(self, data_dir: str = "data/vectors"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    # ── 写入 ──
    def create(self, name: str, ids: List[str], embeddings: np.ndarray,
               metadatas: List[dict], documents: List[str]):
        """创建/覆盖一个集合."""
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        np.save(os.path.join(self.data_dir, f"{name}_embs.npy"), embeddings)
        with open(os.path.join(self.data_dir, f"{name}_ids.json"), "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)
        with open(os.path.join(self.data_dir, f"{name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadatas, f, ensure_ascii=False, default=str)
        with open(os.path.join(self.data_dir, f"{name}_docs.json"), "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False)

    # ── 读取 ──
    def _load(self, name: str):
        """加载集合数据. 返回 (ids, embeddings, metadatas, documents)."""
        embs = np.load(os.path.join(self.data_dir, f"{name}_embs.npy")).astype(np.float32)
        with open(os.path.join(self.data_dir, f"{name}_ids.json"), "r", encoding="utf-8") as f:
            ids = json.load(f)
        # meta/docs 可能不存在 (旧版兼容)
        meta_path = os.path.join(self.data_dir, f"{name}_meta.json")
        metas = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metas = json.load(f)
        docs_path = os.path.join(self.data_dir, f"{name}_docs.json")
        docs = None
        if os.path.exists(docs_path):
            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
        return ids, embs, metas, docs

    def exists(self, name: str) -> bool:
        return os.path.exists(os.path.join(self.data_dir, f"{name}_embs.npy"))

    def count(self, name: str) -> int:
        if not self.exists(name):
            return 0
        embs = np.load(os.path.join(self.data_dir, f"{name}_embs.npy"), mmap_mode="r")
        return embs.shape[0]

    def get(self, name: str, include: List[str] = None):
        """模拟 chromadb 的 get() 接口."""
        ids, embs, metas, docs = self._load(name)
        result = {"ids": ids}
        if include:
            if "embeddings" in include:
                result["embeddings"] = embs.tolist()
            if "metadatas" in include:
                result["metadatas"] = metas or [{}] * len(ids)
            if "documents" in include:
                result["documents"] = docs or [""] * len(ids)
        return result

    # ── 查询 ──
    def query(self, name: str, query_embeddings: List[List[float]],
              n_results: int = 10) -> dict:
        """余弦相似度检索. 返回 chromadb 兼容格式."""
        ids, embs, metas, docs = self._load(name)

        queries = np.array(query_embeddings, dtype=np.float32)
        # L2 normalize queries
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        queries = queries / q_norms

        # 余弦相似度 = queries @ embs.T
        scores = queries @ embs.T  # (n_queries, n_docs)

        result_ids = []
        result_distances = []
        result_metadatas = []
        result_documents = []

        for i in range(len(queries)):
            top_k = min(n_results, len(ids))
            top_idx = np.argpartition(-scores[i], top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-scores[i][top_idx])]

            result_ids.append([ids[j] for j in top_idx])
            # chromadb 用 distance (1 - cosine)
            result_distances.append([float(1.0 - scores[i][j]) for j in top_idx])
            if metas:
                result_metadatas.append([metas[j] for j in top_idx])
            if docs:
                result_documents.append([docs[j] for j in top_idx])

        return {
            "ids": result_ids,
            "distances": result_distances,
            "metadatas": result_metadatas if metas else None,
            "documents": result_documents if docs else None,
        }


# ── 全局单例 ──
_store: Optional[NumpyVectorStore] = None


def get_store(data_dir: str = "data/vectors") -> NumpyVectorStore:
    global _store
    if _store is None:
        _store = NumpyVectorStore(data_dir)
    return _store
