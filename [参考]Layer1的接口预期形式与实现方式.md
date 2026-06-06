# Layer 1 — 评论问答引擎 · 接口与实现

## 1. 定位

无状态的问答服务。给定一个问题，从评论库中找到最相关的证据并生成回答。

**不懂场景、不记对话、不做决策。** 只负责检索 + 生成。

---

## 2. 接口

### 主接口

```python
def ask(
    query: str,                          # 自然语言问题
    filters: dict | None = None,          # 可选过滤，如 {"room_type": "花园大床房", "date_range": ["2024-03","2025-03"]}
    strategy: str = "hybrid",             # "simple" | "hybrid" | "semantic"
    top_k: int = 10
) -> dict:
    """
    返回:
    {
        "answer": str,                    # 自然语言回答，含 [[ref:id]] 引用标记
        "sources": [                      # 引用的评论原文
            {"id": "xxx", "content": "...", "score": 4.2, "room_type": "...", "publish_date": "..."},
            ...
        ],
        "metrics": {                      # 诊断信息
            "retrieval_count": 25,
            "reranked_count": 10,
            "latency_breakdown": {"retrieval": 0.56, "rerank": 0.55, "generation": 3.2}
        }
    }
    """
```

### 辅助接口

```python
def retrieve(
    query: str,
    filters: dict | None = None,
    strategy: str = "hybrid",
    top_k: int = 10
) -> list[dict]:
    """只检索不生成。返回 Top-N 评论列表。用于 Layer 2 自行组装时。"""

def embed(texts: list[str], text_type: str = "document") -> list[list[float]]:
    """批量文本向量化。"""
```

---

## 3. 内部管线

```
query → [查询理解] → [五路并行检索] → [RRF融合 Top-100] → [重排序 Top-10] → [LLM生成] → answer
```

### 3.1 查询理解

- `jieba` 分词 → BM25 查询词
- `text-embedding-v4` → 1024 维语义向量
- 可选：HyDE 生成假评论再向量化

### 3.2 五路并行检索（各路 Top-25）

| 路 | 技术 | 索引目标 | 特点 |
|----|------|---------|------|
| BM25 | jieba + 自建倒排 | 评论文本 | 关键词精准匹配 |
| 向量语义 | text-embedding-v4 → ChromaDB | 评论向量库 | 语义匹配，支持 `filter` |
| 反向Query | 同上 | 反向Query向量库 (6441条) | "搜问题找答案" |
| HyDE | LLM生成假评论 → 向量化 → ChromaDB | 评论向量库 | 弥补短查询 vs 长文档语义鸿沟 |
| 类别摘要 | ChromaDB | 14类预生成摘要 | 概述型查询全局视野 |

### 3.3 RRF 加权融合

```python
def rrf_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion。同文档在不同路中出现，RRF 分数叠加。"""
    scores = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            doc_id = item["id"]
            if doc_id not in scores:
                scores[doc_id] = {"id": doc_id, "content": item["content"], "score": 0.0, "methods": []}
            scores[doc_id]["score"] += 1.0 / (k + rank + 1)
            scores[doc_id]["methods"].append(item["retrieval_method"])
    return sorted(scores.values(), key=lambda x: x["score"], reverse=True)
```

### 3.4 重排序 + 综合打分

- **Qwen3-Rerank**（Cross-Encoder）：Query + 文档拼接，输出 `relevance_score`
- **综合打分**（线性加权）：
  `final = 0.4 × relevance + 0.25 × quality + 0.05 × length_norm + 0.05 × useful_norm + 0.05 × review_norm + 0.2 × recency(half-life)`

### 3.5 LLM 生成

- 上下文组装：系统提示词 + Top-10 评论 + 相关类别摘要 + 历史对话(如有)
- 模型：DashScope Qwen-Plus，流式输出
- 引用格式：回复中内嵌 `[[ref:id1,id2,id3]]` 标记

---

## 4. 实现方式

### 4.1 文件结构

```
engine/
├── client.py          # 主入口，ask() / retrieve() / embed()
├── bm25.py            # BM25Retriever: build() / search() / load() / save()
├── vector_store.py    # ChromaDBStore: insert() / search()
├── reverse_query.py   # ReverseQueryStore: ChromaDB 封装，索引反向Query
├── summary_store.py   # SummaryStore: ChromaDB 封装，索引14类摘要
├── hyde.py            # HyDeGenerator: LLM生成假评论 → 检索
├── fusion.py          # rrf_fusion()
├── reranker.py        # Reranker: DashScope Rerank API 封装 + 综合打分
├── generator.py       # Generator: 上下文组装 + LLM 流式生成
└── diagnostics.py     # 延迟统计 + 各路召回数记录
```

### 4.2 数据准备（首次运行）

```python
# 1. 从 enriched_comments.csv 加载评论
# 2. 构建 BM25 倒排索引 → 保存为 inverted_index.pkl
# 3. 对每条评论调用 text-embedding-v4 → 写入 ChromaDB collection "comments"
# 4. 对每条反向Query调用 text-embedding-v4 → 写入 ChromaDB collection "reverse_queries"
# 5. 对14个类别摘要调用 text-embedding-v4 → 写入 ChromaDB collection "summaries"
```

### 4.3 关键依赖

```
jieba, chromadb, pandas, pyyaml
dashscope (TextEmbedding + TextReRank)
openai (DashScope compatible endpoint → Qwen-Plus)
```

### 4.4 配置项（config.yaml）

```yaml
embedding:
  model: "text-embedding-v4"
  dimension: 1024
  batch_size: 10

reranker:
  model: "gte-rerank"

retrieval:
  hybrid:
    top_k_per_path: 25
  rerank_top_k: 10
  rrf_k: 60

llm:
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"
```

---

## 5. 设计原则

- **无状态**：每次 `ask()` 独立，不保留任何会话记忆
- **可诊断**：`metrics` 返回各路召回数 + 延迟，供 Layer 2 或调试使用
- **可插拔**：HyDE 默认关闭（`strategy="hybrid"` 不含 HyDE），`strategy="hybrid_with_hyde"` 开启
- **策略可选**：支持 `simple`（仅 BM25）、`hybrid`（BM25+Vector+RQ+Summary）、`semantic`（仅向量）
