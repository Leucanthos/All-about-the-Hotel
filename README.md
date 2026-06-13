# 宿说 — 酒店评论智能顾问

基于真实住客评论的多模态检索与问答系统。覆盖 6400+ 条广州花园酒店真实评论。

## 项目结构

```
Project/
├── README.md                           ← 本文件
├── pyproject.toml                      ← 包配置 (pip install -e . → hotel 命令)
├── hotel/                              ← CLI 包
│   ├── __init__.py
│   ├── __main__.py                     ← CLI 入口 (hotel 命令)
│   └── demo.py                         ← 终端交互 Demo
├── configure.py                        ← 环境配置 & 数据准备
│
├── scripts/                            ← 核心代码
│   ├── _shared/                        ← 共享基础设施
│   │   ├── store.py                    ←   NumPy 向量存储
│   │   ├── data.py                     ←   统一数据路径
│   │   └── text.py                     ←   文本处理工具
│   │
│   ├── leuca/   [SU Leuca]             ← Chinese-CLIP 多模态检索
│   │   └── multimodal/                 ←   engine + api + build
│   │
│   └── yuhao/   [SUN Yuhao]            ← Agentic RAG + SFT
│       ├── agentic/                    ←   engine + api + interfaces
│       └── sft/                        ←   build + train
│
├── data/                               ← 永久数据 (不入 git)
│   ├── hotel_reviews_table.parquet
│   ├── category_summaries.json
│   ├── images/  split/  vectors_np/
│
└── refs/                               ← 课程参考文档
```

## Credit

| 模块 | 贡献者 | 内容 |
|------|--------|------|
| `scripts/leuca/` | **SU Leuca** (苏乐茶) | 多模态检索引擎、统一 API、Web Demo、评估体系、微调实验 |
| `scripts/yuhao/` | **SUN Yuhao** (孙宇昊) | Agentic RAG 流水线、SFT 数据集构建、QLoRA 训练、接口适配 |

> 实验和微调脚本见 `scripts/leuca/__arch_research/` 和 `scripts/yuhao/sft/`。

## 快速开始

```bash
# 1. 安装 CLI
pip install -e .

# 2. 启动
hotel
```

> 环境配置（模型下载、索引构建、GPU 加速）见 `python configure.py --help`。

```python
# 多模态检索 (Leuca)
import sys; sys.path.insert(0, 'scripts')
from leuca.multimodal.api import MultimodalAPI
api = MultimodalAPI()
api.prompt2image("游泳池干净吗", topK=5)   # 文本 → 图片

# Agentic RAG 问答 (Yuhao)
import sys; sys.path.insert(0, 'scripts')
from yuhao.agentic.rag import answer_question
print(answer_question("早餐怎么样？")["answer"])
```

## 环境

| 用途 | Python | 环境 |
|------|--------|------|
| 推理 (OpenVINO GPU) | 3.10+ | `agentic` |
| 训练 (Intel Arc XPU) | 3.11 | `agentic_xpu` |

> **注意**: 大文件 (模型权重、图片、向量索引、中间产物) 不入 git。通过 `setup.py` 和对应脚本重新生成。

## 核心发现

| 指标 | 值 | 说明 |
|------|-----|------|
| Relaxed Hit@10 | 51.5% | 同房型检索成功率 |
| Strict Hit@10 | 5.4% | 精确图片匹配 (1200 池中) |
| 最佳文本形式 | 房型名 | MRR 是完整评论的 2.5x |
| GPU 推理加速 | 23-34x | OpenVINO on Intel Arc B390 |
| LTR 重排序 | Hit@5 +6pp | 最有效的算法拓展 |

## CLI 工具

```bash
pip install -e .                    # 一次性安装 → 全局可用 hotel 命令

hotel                               # 终端交互 Demo
hotel advise "游泳池干净吗"          # 单次场景顾问查询
```

> 也可用 `python -m hotel` 运行（无需 pip install）。

## 终端 Demo

```bash
hotel                    # 终端交互 (默认)
hotel demo               # 同上
hotel advise "查询内容"   # 单次查询
```

交互式场景顾问：输入出行场景或具体问题，流式展示识别→检索→研判全过程。

