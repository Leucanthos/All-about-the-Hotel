# Agentic RAG + SFT 优化模块

本目录是基于 `llm course/Exp3/exp3.ipynb` 及其数据产物继续做的优化工作，主要覆盖课程参考方向：

- 方向 18: Agentic RAG。增加检索策略选择、证据评估、低置信度 query 改写和二次检索。
- 方向 16: SFT 后训练数据构建。基于反向查询、真实评论和分类摘要，构造“先证据、再结构化回答”的酒店问答微调数据。

## 目录结构

```text
agentic_rag_sft/
  agentic_rag.py          # Agentic RAG 主逻辑
  build_sft_dataset.py    # SFT 数据集构建脚本
  cli.py                  # 统一命令行入口
  interfaces.py           # 对齐 Layer1/Layer2 的接口适配层
  run_demo.py             # smoke test
  train_sft_lora.py       # GPU 环境下的 QLoRA/Unsloth 训练脚本
  关键提示词.md

data/
  filtered_comments.csv      # exp3.ipynb/Exp3 产物: 清洗后的评论
  reverse_queries.csv        # exp3.ipynb/Exp3 产物: 反向 Query
  category_summaries.json    # exp3.ipynb/Exp3 产物: 分类摘要
  sft/
    sft_train.jsonl
    sft_val.jsonl
    sft_preview.json
```

其中 `data/` 是仓库级数据目录，代码目录中不再额外放数据文件。

## 和 exp3.ipynb 的关系

`exp3.ipynb` 原本已经完成了评论知识库构建与 RAG 基础组件，包括：

- 清洗评论数据: `filtered_comments.csv`
- 反向查询数据: `reverse_queries.csv`
- 分类摘要: `category_summaries.json`
- BM25 倒排索引和 Chroma 向量库等检索资产

本模块目前默认复用前三个轻量数据文件，因此可以直接在仓库中运行。BM25 pkl 和 ChromaDB 资产没有一并提交，原因是当前 Agentic RAG 版本使用轻量本地检索来做 smoke test 和接口对齐；如果后续要接回 `exp3.ipynb` 的完整 BM25/Chroma 检索，可以在 `interfaces.retrieve()` 或 `HotelCorpus.search()` 中替换检索后端，外部接口不需要改。

## Layer1 接口对齐

参考 `[参考]Layer1的接口预期形式与实现方式.md`，本模块在 `interfaces.py` 中提供：

```python
from interfaces import ask, retrieve, embed

ask(query, filters=None, strategy="hybrid", top_k=10) -> dict
retrieve(query, filters=None, strategy="hybrid", top_k=10) -> list[dict]
embed(texts, text_type="document") -> list[list[float]]
```

`ask()` 返回字段包括：

- `answer`: 结构化回答
- `sources`: 评论证据列表
- `metrics`: 检索数量、证据评分、Agent 轨迹、耗时等诊断信息

因此这部分工作可以作为 Layer1 的增强问答引擎接入。和参考文档相比，它的主要优化点是 `metrics.agent_trace` 中包含了 plan、evaluate、rewrite/retry 等 Agentic RAG 过程。

## Layer2 接口对齐

参考 `[参考]Layer2的接口预期形式与实现方式.md`，`interfaces.py` 也提供了一个轻量 `advise()`：

```python
from interfaces import advise

advise(user_input, session_id="default") -> dict
```

返回字段包括：

- `scene`
- `confidence`
- `dimensions`
- `verdict`
- `reply`
- `sources`
- `metrics`

这个 `advise()` 主要用于和同事的 Layer2 Agent 对齐接口。更完整的场景识别、多轮记忆和维度矩阵可以由同事的 Layer2 继续实现；他们只需要调用本模块的 `ask()`，即可拿到回答、证据和 Agent 轨迹。

## 快速运行

在仓库根目录执行：

```powershell
cd C:\Users\syh\jupyter\All-about-the-Hotel\agentic_rag_sft

# 方向 16: 重新构建 SFT 数据，默认写入 ../data/sft
py .\cli.py q16 build-sft --max-samples 800

# 方向 18: 运行 Agentic RAG 问答，并展示检索轨迹
py .\cli.py q18 ask "早餐和前台服务怎么样？有哪些常见吐槽？" --show-trace

# smoke test: 小规模构建写入 ../data/demo_run，再跑一次 Agentic RAG
py .\run_demo.py
```

如果配置了 DashScope 或 OpenAI 兼容接口，可以让 Agent 调用 LLM 生成最终回答：

```powershell
$env:DASHSCOPE_API_KEY="你的 key"
py .\cli.py q18 ask "这家酒店适合带老人小孩入住吗？房间和交通怎么样？" --use-llm
```

未配置 API key 时，系统会使用本地模板回答，仍可展示检索、证据评分和重试逻辑。

## 已生成结果

当前已补充的 SFT 结果文件位于：

- `data/sft/sft_train.jsonl`
- `data/sft/sft_val.jsonl`
- `data/sft/sft_preview.json`

这些样本采用 `messages` 和 `conversations` 双格式，既可以用于 OpenAI 风格对话数据，也可以用于常见国产大模型微调流程。

## 后续接入建议

如果同事已经实现了完整 Layer1 检索服务，可以把 `interfaces.ask()` 内部的 `answer_question()` 替换为同事的 Layer1 `ask()`，保留本模块的证据评估、query 改写和 retry 逻辑。

如果同事负责 Layer2 场景顾问，可以直接调用：

```python
from interfaces import ask

result = ask("带老人小孩入住合适吗？房间和交通怎么样？")
```

然后把 `result["answer"]`、`result["sources"]`、`result["metrics"]["agent_trace"]` 拼入 Layer2 的综合研判和结构化回复中。
