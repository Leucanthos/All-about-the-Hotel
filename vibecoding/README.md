# 多模态召回 — 酒店评论跨模态检索

## 目录结构

```
vibecoding/
├── README.md                   ← 本文件
├── analysis_report.md          ← 最终交付报告
│
├── src/                        ← 生产环境
│   ├── multimodal_api.py       ← 核心接口
│   ├── numpy_vector_store.py   ← 向量存储
│   └── build_index.py          ← 索引构建
│
└── research/                   ← 论证材料
    ├── test_and_analysis.py    ← 测评体系
    ├── finetune.py             ← 微调实验框架
    ├── experiments.py          ← LTR/分类器迭代实验
    ├── test_results.json       ← 测试结果
    └── experiment_log.json     ← 实验日志
```

## 快速使用 (生产环境)

```python
from src.multimodal_api import MultimodalAPI

api = MultimodalAPI()
api.prompt2image("游泳池干净吗", topK=5)
api.image2text("data/images/img_0.jpg", topK=10)
```

## 索引构建

```bash
python src/build_index.py --images 1200 --texts 2000
```

## 论证 & 测评

```bash
# 测评
python research/test_and_analysis.py

# 微调实验
python research/finetune.py --experiment baseline --epochs 10
```

## 环境

- **推理**: Python 3.14 (agentic env) + OpenVINO GPU
- **训练**: Python 3.11 (agentic_xpu env) + Intel Arc B390 XPU
