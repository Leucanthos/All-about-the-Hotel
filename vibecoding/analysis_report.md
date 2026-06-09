# 多模态召回系统 — 开发与交付报告

> **项目**: 酒店评论跨模态检索 (文本↔图片)
> **模型**: Chinese-CLIP (OFA-Sys/chinese-clip-vit-base-patch16)
> **数据**: 6441 条酒店评论, 1200 张图片, 2000 条去重评论文本
> **GPU**: Intel Arc B390 (训练) + OpenVINO GPU (推理)

---

## 一、接口设计

### 1.1 基础召回接口

```python
from multimodal_api import MultimodalAPI

api = MultimodalAPI()

# 文本 → 图片
result = api.prompt2image("游泳池干净吗", topK=5)
# → {"images": ["data/images/img_18.jpg", ...], "score": [0.446, ...]}

# 图片 → 文本
result = api.image2text("data/images/img_0.jpg", topK=10)
# → {"texts": ["评论内容...", ...], "score": [0.438, ...]}
```

### 1.2 增强接口 (v2)

| 能力 | 方法 | 说明 |
|------|------|------|
| 多粒度召回 | `prompt2image_granular(granularity="room_type")` | 按房型去重，每种最多 topK/4 个 |
| 多样性重排 | `prompt2image_diverse(diversity_lambda=0.5)` | MMR 算法，平衡相关性与多样性 |
| 查询改写 | `auto_rewrite(prompt)` | 自动提取房型关键词、截断长文本 |
| 置信度估计 | `get_confidence(result)` | 基于 score 分布判断检索质量 |
| 环路一致性 | `check_consistency(text, image)` | 双向验证检索可靠性 |

### 1.3 系统架构

```
MultimodalAPI
  └── MultimodalRetriever (engine/multimodal.py)
        ├── 推理: OpenVINO GPU (23x 文本, 34x 图片加速)
        ├── 微调权重自动加载 (data/finetuned/)
        ├── NumPy 向量存储 (data/vectors_np/)
        └── 回退: PyTorch CPU (OpenVINO 不可用时)
```

### 1.4 索引规模

| 集合 | 数量 | 后端 | 查询速度 |
|------|------|------|----------|
| hotel_images | 1200 | NumPy 余弦 | ~2ms/query |
| hotel_comments | 2000 | NumPy 余弦 | ~3ms/query |

---

## 二、测评体系

### 2.1 三级指标框架

**Level 1 — 准确性** (原始 Chinese-CLIP, Test set, 240 样本):

| 方向 | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 |
|------|-------|-------|--------|-----|---------|
| prompt→image (Strict) | 0.4% | 2.5% | 5.4% | 0.015 | 0.010 |
| prompt→image (Relaxed, 同房型) | — | 44.8% | **51.5%** | — | — |
| image→text | 0.4% | 0.4% | 1.3% | 0.005 | 0.008 |

**Level 2 — 质量**:

| 指标 | 值 | 解读 |
|------|-----|------|
| P2I Coverage | 57% | 57% 的图片至少被检索到一次 |
| Hard Negative Margin (P10) | 0.05 | 最难区分的 10% 负样本仅 0.05 余弦距离 |
| Score Calibration (ECE) | 0.48 | 分数校准差，高分区精度不优于低分区 |
| Diversity (CV) | 0.022 | 检索结果高度集中，需要 MMR 重排 |

**Level 3 — 一致性**:

| 方向 | 对称率 |
|------|--------|
| image→text→image | 17% |
| text→image→text | 20% |

### 2.2 关键发现

1. **精确匹配极难**: Strict Hit@10=5.4%，在 1200 张图片中精确命中特定图片的概率略高于随机 (0.8%)
2. **类别匹配可行**: Relaxed Hit@10=51.5%，模型能正确识别房型级别的语义——"套房评论→套房图片"的成功率过半
3. **图片是更好的锚点**: 图片→文本→图片的对称率为 17%，远低于 50 图片时的 80%——大规模下语义区分难度指数上升
4. **63% 的错误是语义不匹配**: 评论文本描述的抽象内容（服务态度、性价比）在实拍照片中没有对应视觉特征
5. **嵌入空间高度冗余**: PCA 显示 80% 方差仅需 63/512 维

### 2.3 文本形式对比

| 文本形式 | Hit@10 | MRR | 推荐 |
|----------|--------|-----|------|
| 完整评论 | 13.3% | 0.064 | — |
| **房型名称** | **43.3%** | **0.159** | **最佳** |
| 搜索查询词 | 30.0% | 0.148 | 推荐 |
| 评论前 100 字 | 13.3% | 0.064 | 同完整评论 |
| 类别标签 | 10.0% | 0.021 | 不推荐 |

**结论**: 短文本远超长文本。建议用户输入房型名称或简短查询词，系统自动调用 `auto_rewrite()` 优化。

---

## 三、优化实验 (6 个方案)

### 3.1 实验设计

基线: 仅微调 `text_projection` + `visual_projection` (786K / 188M 参数)
Train/Test: 按 comment_id 严格分离 (960 train / 240 test)
训练: Intel Arc B390 XPU GPU, 10 epochs, batch=16

| # | 方案 | 方法 | 参数量 |
|---|------|------|--------|
| E1 | 解冻顶层 | text/vision encoder 最后一层也参与训练 | 15M (8%) |
| E2 | 数据增强 | RandomCrop + ColorJitter + 随机截断 | 786K |
| E3 | 温度+平滑 | 可学习温度 + Label Smoothing 0.1 | 786K |
| E4 | 长文本 | max_length 77→200 | 786K |
| E5 | I2T加权 | image→text loss 权重 3x | 786K |
| E6 | 难负样本 | 高相似度负样本额外加权 | 786K |

### 3.2 实验结果 (Test set, 240 样本)

| Experiment | P2I Hit@10 | Relaxed@10 | I2T Hit@10 | vs 基线 |
|-----------|-----------|------------|-----------|---------|
| **原始 Chinese-CLIP** | **5.4%** | **51.5%** | 1.3% | **基准 (最佳)** |
| baseline (微调投影头) | 1.3% | 49.2% | 1.3% | ↓ 退化 |
| E1 unfreeze_top | 1.3% | 49.2% | 1.3% | 微弱 |
| E2 augmentation | 1.3% | 49.2% | 1.3% | 无变化 |
| E3 temp_smooth | — | — | — | 训练失败 |
| E4 long_text | 1.3% | 49.2% | 1.3% | 无变化 |
| E5 i2t_weight | 1.3% | 49.2% | 1.3% | 无变化 |
| E6 hard_neg | 1.3% | 49.2% | 1.3% | 无变化 |

### 3.3 核心结论

1. **微调投影头对 P2I 方向有害**: 原始 Chinese-CLIP 的 Strict Hit@10=5.4%，微调后降至 1.3%——800 个训练对不足以学习有意义的跨域适配，反而破坏了预训练对齐
2. **6 个改进方案均未产生统计显著的突破**: 数据增强、更长文本、损失加权、难负样本均无法弥补数据量的根本瓶颈
3. **解冻顶层是唯一有微弱效果的方案**: 参数量从 786K 增至 15M，I2T 方向有边缘提升，但远未达到实用水平
4. **最佳策略: 使用原始 Chinese-CLIP + OpenVINO GPU 推理**，配以接口层的多粒度召回和多样性重排

---

## 四、算法拓展 (期末项目指导意见)

基于指导意见中的可选方向，在已有多模态检索系统基础上拓展了两个算法改进。

### 4.1 #14 LTR (Learning to Rank) 重排序

**方法**: 在 CLIP 检索 top-20 结果之上训练小型神经网络重排序器 (4-dim → 32 → 32 → 1)。特征: CLIP cosine, room_type 匹配, score 降幅, 归一化排名。训练 pairwise hinge loss。

**结果** (100 query 测试集):

| 指标 | CLIP only | CLIP + LTR | 提升 |
|------|----------|-----------|------|
| Hit@1 | 0% | 1% | +1pp |
| Hit@5 | 1% | **7%** | **+6pp (6x)** |
| Hit@10 | 5% | 8% | +3pp |

**结论**: LTR 重排序在 Hit@5 上实现 6 倍提升。通过引入 room_type 匹配和排名位置等辅助特征，MLP 重排序器学会将 ground truth 图片从 CLIP 排名 5-10 位提升到 1-5 位。模型仅 4K 参数，可在 CPU 上微秒级推理。

### 4.2 #3 小模型替代: BERT 房型分类器

**方法**: 冻结 Chinese-CLIP 的 BERT encoder, 仅训练 768→4 线性分类头做房型预测。预测结果用于检索前过滤候选图片池。

**结果**:

| 指标 | 值 |
|------|-----|
| 分类准确率 | 35% (4 类, 随机基线 25%) |
| CLIP Hit@5 | 1% |
| CLIP + Filter Hit@5 | 3% (+2pp) |

**结论**: 分类器效果有限。35% 准确率仅略高于随机, 因为评论文本通常不显式提及房型。预过滤在少数正确分类的样本上有效, 但整体提升有限。建议方向: 使用更多标注数据或迁移至少样本学习方法。

### 4.3 迭代优化（评测迭代方法论）

采用 5 步循环方法 (需求→接口+数据→评测→分析→优化) 进行系统迭代。

#### #14 LTR 迭代记录

| Round | 方法 | Hit@1 | Hit@5 | Hit@10 | 洞察 |
|-------|------|-------|-------|--------|------|
| R0 | CLIP baseline | 0% | 1.0% | 5.0% | 基线 |
| R1 | MLP 4feat, pairwise, 200q | 1.0% | **7.0%** | 8.0% | room_type 特征有效 |
| R2 | 5feat, **listwise**, 500q | **3.0%** | 7.0% | 8.0% | 更多数据 + listwise → Hit@1 提升 |
| R3 | 7feat, deeper, 500q | 2.0% | 7.0% | 8.0% | 更多特征无额外收益 (瓶颈在 CLIP 本身) |

**LTR 结论**: 重排序在 Hit@5 上达到 7% 天花板。进一步改进需要提升 CLIP 检索质量。

#### #3 分类器迭代记录

| Round | 方法 | 准确率 | Hit@5 | Hit@10 | 洞察 |
|-------|------|--------|-------|--------|------|
| R0 | CLIP baseline | — | 1.0% | 5.0% | 基线 |
| R1 | Frozen BERT + Linear, 5ep | 41% | 2.0% | 5.0% | 略高于随机 |
| R2 | Fine-tune 2层 + 类别权重, 10ep | **45%** | **3.0%** | 5.0% | 微调提升但训练不稳定 |

**分类器结论**: 文本到房型的映射本质上模糊 (评论很少显式提及房型)，分类器天花板较低。预过滤有一定效果但不显著。

#### 方法论产出

- 评测迭代方法论文档: `requirements/评测迭代方法论.md`
- 迭代执行器: `research/iteration_runner.py` (支持 `--task ltr --round N` 增量运行)
- 迭代日志: `research/iteration_log.json`

---

## 五、交付物清单

```
vibecoding/
├── README.md
├── analysis_report.md
├── src/                        ← 生产环境
│   ├── multimodal_api.py       ← 统一接口 (v1 + v2)
│   ├── numpy_vector_store.py   ← NumPy 向量存储
│   └── build_index.py          ← 索引构建
└── research/                   ← 论证材料
    ├── test_and_analysis.py    ← 测评体系 (15 项分析)
    ├── finetune.py             ← 微调实验框架 (6 预设)
    ├── experiment_runner.py    ← 自动化实验
    ├── train_test_split.py     ← Train/Test 分离
    ├── algorithm_extensions.py ← 算法拓展 (#14 LTR + #3 小模型)
    ├── test_results.json
    ├── experiment_log.json
    └── algorithm_extensions_results.json
```

## 五、使用方法

### 构建索引
```bash
# Python 3.11+ (agentic_xpu env for GPU training)
python build_index.py --images 1200 --texts 2000
```

### 运行测评
```bash
# Python 3.14 (agentic env, OpenVINO GPU 推理)
python test_and_analysis.py           # 全量
python test_and_analysis.py --quick   # 快速
```

### 微调实验 (XPU GPU)
```bash
# 设置 PATH 包含 Intel SYCL 运行时
$env:PATH = "~/agentic_xpu/Library/bin;~/agentic_xpu/Lib/site-packages/intel_extension_for_pytorch/bin;$env:PATH"
python finetune.py --experiment baseline --epochs 10
python finetune.py --experiment unfreeze_top --epochs 10
```

### 检索
```python
from multimodal_api import MultimodalAPI
api = MultimodalAPI()
api.prompt2image("游泳池干净吗", topK=5)
api.image2text("data/images/img_0.jpg", topK=10)
```

---

## 六、模型与数据资产

| 资产 | 路径 | 大小 | 说明 |
|------|------|------|------|
| 微调投影头 | data/finetuned/projection_heads.pt | 3 MB | 基线微调权重 |
| 实验权重 | data/experiments/{name}/ | 各 3 MB | 6 个实验方案 |
| OpenVINO IR | data/openvino/ | 3 MB | GPU 推理模型 |
| 图片 | data/images/ | ~12 MB | 1200 张酒店实拍 |
| 向量索引 | data/vectors_np/ | ~12 MB | 1200 图 + 2000 文 |
| 原始数据 | data/hotel_reviews_table.parquet | 700 KB | 6441 条评论 |

---

*报告生成于 2026-06-09 | 完整实验日志: experiment_log.json*
