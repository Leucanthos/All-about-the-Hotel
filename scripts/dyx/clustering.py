"""dyx 聚类分析模块 — TF-IDF + UMAP + HDBSCAN 评论主题聚类.

从 clustering_categories.ipynb 提取重构，适配多酒店 parquet 数据。

用法:
    from dyx.clustering import CommentClusterer
    clusterer = CommentClusterer()
    clusterer.fit(comments)  # List[str]
    labels = clusterer.labels
    # 或加载预计算结果
    clusterer.load("data/cluster_assignments.json")
"""

import json, os, re, pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class CommentClusterer:
    """评论主题聚类器 — TF-IDF → UMAP → HDBSCAN."""

    def __init__(self, n_components: int = 5, min_cluster_size: int = 30,
                 min_samples: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.random_state = random_state
        self._vectorizer = None
        self._umap_model = None
        self._cluster_model = None
        self.labels: Optional[np.ndarray] = None
        self.cluster_metadata: Dict = {}
        self.comment_ids: List[str] = []

    def fit(self, texts: List[str], doc_ids: Optional[List[str]] = None,
            max_features: int = 2000):
        """对评论列表执行全流程聚类.

        Args:
            texts: 评论文本列表
            doc_ids: 对应的文档ID (可选)
            max_features: TF-IDF 最大特征数
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        import umap, hdbscan
        import jieba

        print(f"[cluster] Fitting {len(texts)} docs...")

        # 1. TF-IDF (用 jieba 分词)
        def tokenizer(text: str):
            return [w for w in jieba.lcut(text)
                    if len(w) > 1 and re.match(r'[\u4e00-\u9fff]', w)]

        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenizer, max_features=max_features,
            max_df=0.85, min_df=5, norm="l2"
        )
        tfidf = self._vectorizer.fit_transform(texts)
        print(f"  TF-IDF: {tfidf.shape}")

        # 2. UMAP 降维
        self._umap_model = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            metric="cosine", n_neighbors=15,
        )
        embedding = self._umap_model.fit_transform(tfidf)
        print(f"  UMAP: {embedding.shape}")

        # 3. HDBSCAN 聚类
        self._cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            gen_min_span_tree=True,
        )
        self.labels = self._cluster_model.fit_predict(embedding)
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = sum(1 for l in self.labels if l == -1)
        print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise ({n_noise/len(texts)*100:.0f}%)")

        self.comment_ids = doc_ids or [str(i) for i in range(len(texts))]
        self._build_metadata(texts)
        return self

    def _build_metadata(self, texts: List[str]):
        """为每个聚类生成关键词和统计."""
        import jieba
        from collections import Counter

        cluster_texts = {}
        for label, text in zip(self.labels, texts):
            if label == -1:  # noise
                continue
            if label not in cluster_texts:
                cluster_texts[label] = []
            cluster_texts[label].append(text)

        self.cluster_metadata = {}
        for label, ctexts in cluster_texts.items():
            # 提取关键词
            words = []
            for t in ctexts:
                words.extend(w for w in jieba.lcut(t)
                            if len(w) > 1 and re.match(r'[\u4e00-\u9fff]', w))
            top_kw = [w for w, _ in Counter(words).most_common(10)]

            self.cluster_metadata[int(label)] = {
                "cluster_id": int(label),
                "comment_count": len(ctexts),
                "keywords": "、".join(top_kw[:5]),
                "top_keywords": top_kw,
            }

    def get_comment_cluster(self, doc_id: str) -> int:
        """返回某条评论所属的 cluster ID (-1 = noise/未分类)."""
        if doc_id in self.comment_ids:
            idx = self.comment_ids.index(doc_id)
            return int(self.labels[idx]) if self.labels is not None else -1
        return -1

    def save_assignments(self, path: str):
        """保存聚类结果."""
        data = {
            "cluster_metadata": self.cluster_metadata,
            "assignments": dict(zip(self.comment_ids, map(int, self.labels))),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[cluster] Saved assignments to {path}")

    def load(self, path: str):
        """加载预计算聚类结果."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cluster_metadata = data.get("cluster_metadata", {})
        assignments = data.get("assignments", {})
        self.comment_ids = list(assignments.keys())
        self.labels = np.array(list(assignments.values()))
        print(f"[cluster] Loaded {len(self.comment_ids)} assignments from {path}")
        return self


def run_clustering(parquet_path: str, output_path: str, sample: int = 5000):
    """便捷入口: 从 parquet 读取评论 -> 聚类 -> 保存.

    Args:
        parquet_path: 输入的 parquet 文件路径
        output_path: 输出的 JSON 保存路径
        sample: 采样数, None=全量
    """
    df = pd.read_parquet(parquet_path)
    texts = df["comment"].dropna().tolist()
    ids = df["_id"].astype(str).tolist()

    if sample and len(texts) > sample:
        import random
        random.seed(42)
        combined = list(zip(texts, ids))
        sampled = random.sample(combined, sample)
        texts, ids = zip(*sampled)
        texts, ids = list(texts), list(ids)

    print(f"[cluster] Loaded {len(texts)} comments from {parquet_path}")
    clusterer = CommentClusterer()
    clusterer.fit(texts, ids)
    clusterer.save_assignments(output_path)
    return clusterer
