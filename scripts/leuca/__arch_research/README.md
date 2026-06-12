# Archived Research — SU Leuca 实验 & 论证

每种召回/优化方法独立一个文件夹，便于理解各自的技术路线。

## 目录

| 文件夹 | 方法 | 核心脚本 |
|--------|------|----------|
| `finetune/` | Chinese-CLIP 对比微调 (6 组实验) | `finetune.py` |
| `ltr/` | Learning to Rank 重排序 (R0-R6) | `experiments.py` |
| `evaluation/` | 8 维测评体系 | `test_and_analysis.py` |

## 与生产代码的关系

- `finetune/`: 产出 `data/finetuned/projection_heads.pt`，由 `leuca/multimodal/engine.py` 可选加载
- `ltr/`: 产出 `leuca/multimodal/ltr_model.pt`，由 `leuca/multimodal/api.py` 的 `prompt2image_rerank()` 调用
- `evaluation/`: 纯测评，不影响生产代码

## 设计文档

| 文件夹 | 内容 |
|--------|------|
| docs/ | 设计文档 & 最终分析报告 |
