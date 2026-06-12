"""_shared/clusters.py — 聚类感知检索 (dyx 贡献).

基于 TF-IDF + UMAP + HDBSCAN 的评论主题聚类，提供:
  1. 查询 → 相关集群映射 (关键词匹配)
  2. 集群评分提升 (对属于相关集群的文档加分)

用法:
    from _shared.clusters import ClusterBoost
    boost = ClusterBoost.load()
    cluster_ids = boost.match_query("游泳池干净吗")
    # cluster_ids -> [0, 11]  # 设施类、泳池类
"""

import json, os, re
from pathlib import Path
from typing import Dict, List, Optional, Set

from _shared.data import DATA


class ClusterBoost:
    """聚类感知的检索提升器.

    加载预计算的 cluster assignments，根据查询匹配相关集群，
    在 BM25 分数基础上加 cluster_boost.
    """

    def __init__(self):
        self.metadata: Dict[int, Dict] = {}
        self.assignments: Dict[str, int] = {}  # doc_id -> cluster_id

    @classmethod
    def load(cls, path: Optional[str] = None):
        """从 JSON 加载预计算聚类结果."""
        if path is None:
            path = str(DATA / "cluster_assignments.json")
        obj = cls()
        if not os.path.exists(path):
            print(f"[clusters] No assignments found at {path}, skipping")
            return obj
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj.metadata = {int(k): v for k, v in data.get("cluster_metadata", {}).items()}
        obj.assignments = data.get("assignments", {})
        print(f"[clusters] Loaded {len(obj.assignments)} assignments, "
              f"{len(obj.metadata)} clusters from {path}")
        return obj

    def match_query(self, query: str) -> List[int]:
        """返回与查询相关的 cluster ID 列表 (按匹配度排序)."""
        if not self.metadata:
            return []
        scores = []
        for cid, meta in self.metadata.items():
            keywords = meta.get("top_keywords", [])
            # 简单计数: 查询命中几个关键词
            hits = sum(1 for kw in keywords if kw in query)
            if hits > 0:
                scores.append((hits, cid))
        scores.sort(key=lambda x: -x[0])
        return [cid for _, cid in scores]

    def get_cluster_id(self, doc_id: str) -> int:
        """返回文档所属的 cluster (-1 = 未分类)."""
        return int(self.assignments.get(doc_id, -1))

    def boost_score(self, query: str, doc_id: str, base_score: float,
                    boost_factor: float = 1.5) -> float:
        """如果文档属于查询匹配的集群，提升分数."""
        matched_clusters = set(self.match_query(query))
        if not matched_clusters:
            return base_score
        doc_cluster = self.get_cluster_id(doc_id)
        if doc_cluster in matched_clusters:
            return base_score * boost_factor
        return base_score

    def get_cluster_keywords(self, cluster_id: int) -> List[str]:
        """返回集群的关键词列表."""
        meta = self.metadata.get(cluster_id, {})
        return meta.get("top_keywords", [])

    def describe_cluster(self, cluster_id: int) -> str:
        """人类可读的集群描述."""
        meta = self.metadata.get(cluster_id, {})
        if not meta:
            return f"Cluster {cluster_id} (unknown)"
        return f"Cluster {cluster_id}: {meta.get('keywords', '?')} ({meta.get('comment_count', 0)} comments)"

    @property
    def n_clusters(self) -> int:
        return len(self.metadata)


# 全局单例
_cluster_boost: Optional[ClusterBoost] = None


def get_cluster_boost(force_reload: bool = False) -> ClusterBoost:
    """获取全局 ClusterBoost 单例."""
    global _cluster_boost
    if _cluster_boost is None or force_reload:
        _cluster_boost = ClusterBoost.load()
    return _cluster_boost
