# 增强算法模块 (dyx)

本模块提取自 scripts/dyx/lib.py / lib_syh.py, 重构为与 leuca/yuhao 一致的模块化结构.

## 目录结构

`
dyx/
├── __init__.py     ← 包初始化
├── engine.py       ← 核心算法: 意图分类, 查询扩展, 检索编排
├── api.py          ← 公开 API: bm25_search, multi_path_search, expand_query
├── build.py        ← 索引构建工具
├── lib.py          ← 原始完整实现 (保留参考)
├── lib_syh.py      ← 原始完整实现 (保留参考)
├── clustering_categories.ipynb  ← 聚类分析 notebook
├── Database.ipynb  ← 数据库相关 notebook
├── QA-sys.ipynb    ← QA 系统 notebook
├── exp3-syh.ipynb  ← 实验三 notebook
└── hotel-report.tex ← LaTeX 报告
`

## 核心算法贡献

| 算法 | 来源 | 说明 |
|------|------|------|
| **BM25 中文倒排索引** | lib.py InvertedIndex | jieba 分词 + BM25 排序 → 迁移至 _shared/bm25.py |
| **细粒度意图分类** | lib_syh.py detect_intent | 8+ 维度中文关键词 → 迁移至 _shared/router.py |
| **查询扩展** | lib.py xpand_intent | LLM 生成相关子问题 |
| **RRF 多路融合** | lib.py use_results | Reciprocal Rank Fusion → 迁移至 _shared/fusion.py |
| **HyDE 检索** | lib_syh.py _retrieve_hyde | 假设文档嵌入 |
| **反向 Query 检索** | lib.py etrieve_reverse_query | query↔query 匹配 |
| **LTR 重排序** | lib_syh.py ltr_rerank | 学习排序重排 |

## 使用示例

`python
# BM25 检索
from dyx.api import bm25_search
result = bm25_search(\"游泳池干净吗\", top_k=10)

# 多路融合检索
from dyx.api import multi_path_search
result = multi_path_search(\"早餐怎么样\", top_k=10)

# 意图分类
from dyx.api import intent_classify
intent = intent_classify(\"适合带孩子吗\")

# 查询扩展
from dyx.api import expand_query
queries = expand_query(\"房间安静吗\")
`

## 集成说明

- BM25 和 RRF 已提取为 _shared 模块, 可供 leuca/yuhao 共享使用
- 意图分类增强已接入 _shared/router.py 的 classify_intent_detailed()
- 主 workflow 通过 _shared/router.route() 统一调用
