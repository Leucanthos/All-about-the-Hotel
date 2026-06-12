# dyx 研究归档 (__arch_research)

本目录包含 dyx 在开发过程中的探索性工作、原始草稿和实验笔记。
这些文件已从主流水线中重构为模块化代码，保留在此处供追溯和参考。

## 目录内容

| 文件 | 类型 | 说明 | 重构去向 |
|------|------|------|---------|
| `clustering_categories.ipynb` | Notebook | TF-IDF + UMAP + HDBSCAN 聚类分析 | `scripts/dyx/clustering.py` |
| `Database.ipynb` | Notebook | 数据探索与预处理 | — |
| `exp3-syh.ipynb` | Notebook | 实验三开发笔记 | — |
| `QA-sys.ipynb` | Notebook | 问答系统原型 | `_shared/retriever.py` |
| `lib.py` | Python | 原始算法合集 | `_shared/bm25.py`, `_shared/fusion.py` |
| `lib_syh.py` | Python | Yuhao 协作版 | `yuhao/agentic/engine.py` |
| `hotel-report.tex` | LaTeX | 实验报告草稿 | — |
