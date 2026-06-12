"""
BM25 中文倒排索引 — 基于 jieba 分词的 BM25 算法.

从 scripts/dyx/lib.py 提取重构，适配现有 _shared 基础设施。

用法:
    from _shared.bm25 import InvertedIndex
    index = InvertedIndex()
    index.build(documents)       # {doc_id: text, ...}
    results = index.search(query, topk=10)  # [(doc_id, score), ...]
"""

import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class InvertedIndex:
    """基于 BM25 的中文倒排索引."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index: Dict[str, Dict[str, int]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0.0
        self.num_docs = 0
        self.documents: Dict[str, str] = {}

        try:
            import jieba
            jieba.initialize()
            self._jieba = jieba
        except ImportError:
            self._jieba = None

    def tokenize(self, text: str) -> List[str]:
        """jieba 分词 + 去停用词 + 过滤非中英文."""
        text = re.sub(r'\\s+', '', text or '')
        if self._jieba:
            tokens = self._jieba.lcut(text)
        else:
            # fallback: char-level tokens
            tokens = list(text)
        pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9]')
        return [t.lower() for t in tokens
                if t.strip() and not pattern.search(t)]

    def build(self, documents: Dict[str, str]):
        """构建倒排索引."""
        self.documents = documents
        self.num_docs = len(documents)
        total_length = 0

        for doc_id, text in documents.items():
            tokens = self.tokenize(text)
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length

            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                if term not in self.index:
                    self.index[term] = {}
                self.index[term][doc_id] = freq

        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0

    def search(self, query: str, topk: int = 10) -> List[Tuple[str, float]]:
        """BM25 检索."""
        query_tokens = self.tokenize(query)
        if not query_tokens or self.num_docs == 0:
            return []

        idf = {}
        for term in query_tokens:
            if term in self.index:
                df = len(self.index[term])
                idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

        scores: Dict[str, float] = {}
        for term in query_tokens:
            if term not in self.index:
                continue
            for doc_id, tf in self.index[term].items():
                doc_length = self.doc_lengths.get(doc_id, self.avg_doc_length)
                norm_factor = 1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1))
                term_score = idf.get(term, 0) * (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)
                scores[doc_id] = scores.get(doc_id, 0) + term_score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return sorted_docs

    def save(self, filepath: str):
        """序列化保存."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'num_docs': self.num_docs,
                'documents': self.documents,
                'k1': self.k1,
                'b': self.b,
            }, f)

    def load(self, filepath: str):
        """加载序列化索引."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_lengths = data['doc_lengths']
            self.avg_doc_length = data['avg_doc_length']
            self.num_docs = data['num_docs']
            self.documents = data['documents']
            self.k1 = data['k1']
            self.b = data['b']
