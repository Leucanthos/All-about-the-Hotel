# 宿说 — 酒店评论智能顾问

基于真实住客评论的多模态检索系统。支持文本搜图片、图片搜文本，覆盖 6400+ 条广州花园酒店真实评论。

## 项目结构

```
Project/
├── README.md                           ← 本文件
├── engine/multimodal.py                ← 核心检索引擎 (Chinese-CLIP)
│
├── vibecoding/                         ← vibecoding 交付物
│   ├── README.md                       ← 使用指南
│   ├── analysis_report.md              ← 最终交付报告
│   ├── src/                            ← 生产环境
│   │   ├── multimodal_api.py           ← 统一接口 (v1 + v2)
│   │   ├── numpy_vector_store.py       ← NumPy 向量存储
│   │   └── build_index.py              ← 索引构建
│   ├── research/                       ← 论证材料
│   │   ├── test_and_analysis.py        ← 三级指标测评
│   │   ├── finetune.py                 ← 微调实验框架
│   │   ├── experiments.py              ← LTR/分类器迭代实验
│   │   ├── test_results.json           ← 测试结果
│   │   └── experiment_log.json         ← 实验日志
│   └── reqs/                           ← 需求文档
│       ├── 召回要求.md                 ← 接口 + 测评要求
│       ├── 迭代方法论.md               ← 评测迭代方法论
│       └── 实验与优化.md               ← 优化实验记录
│
└── data/                               ← 数据 (不入 git)
    ├── hotel_reviews_table.parquet     ← 原始数据
    ├── images/                         ← 1200 张酒店图片
    ├── vectors_np/                     ← NumPy 向量索引
    ├── finetuned/                      ← 微调权重
    ├── experiments/                    ← 6 组实验权重
    ├── openvino/                       ← GPU 推理模型
    └── split/                          ← Train/Test 划分
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 检查环境
python setup.py --check

# 3. 准备数据: 将 hotel_reviews_table.parquet 放入 data/ 目录

# 4. 下载图片 + 构建索引
python setup.py --index --images 500 --texts 1000

# 5. (可选) GPU 加速
python setup.py --gpu

# 6. 验证
python setup.py
```

```python
# 生产环境检索
import sys; sys.path.insert(0, 'vibecoding/src')
from multimodal_api import MultimodalAPI

api = MultimodalAPI()
api.prompt2image("游泳池干净吗", topK=5)   # 文本 → 图片
api.image2text("data/images/img_0.jpg")   # 图片 → 文本
```

## 环境

| 用途 | Python | 环境 |
|------|--------|------|
| 推理 (OpenVINO GPU) | 3.10+ | `agentic` |
| 训练 (Intel Arc XPU) | 3.11 | `agentic_xpu` |

> **注意**: 大文件 (模型权重、图片、向量索引) 不入 git。通过 `setup.py` 自动下载/构建。

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
python vibecoding/src/cli.py search "游泳池干净吗"          # CLIP 检索
python vibecoding/src/cli.py search "套房" --rerank        # LTR 重排序
python vibecoding/src/cli.py search "大床房" --granular     # 多粒度
python vibecoding/src/cli.py search "装修" --filter         # 分类器预过滤
python vibecoding/src/cli.py reverse 0                      # 图片→文本 (img_0)
python vibecoding/src/cli.py classify "房间很好 装修豪华"    # 房型分类
python vibecoding/src/cli.py stats                          # 索引统计
```

## Web Demo

```bash
python demo.py                             # 启动 → http://localhost:8080
```

4 个 Tab: 文本→图片 / 图片→文本 / 房型分类 / 统计, 4 种检索算法可选。

详见 `vibecoding/analysis_report.md`
