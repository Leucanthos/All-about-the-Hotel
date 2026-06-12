# Evaluation — 多模态检索测评体系

**方法**: 8 维度结构化评测框架

## 测评维度

| # | 维度 | 指标 |
|---|------|------|
| 1 | 基础可用性 | prompt→image, image→text 端到端 |
| 2 | 检索精度 | Hit@K (strict + relaxed) + NDCG |
| 3 | 对称性 | 文本↔图片环路一致性 |
| 4 | 文本形式 | 7 种变体 (完整评论/关键词/房型名/...) |
| 5 | Score 校准 | 分布分析、校准曲线、相对阈值 |
| 6 | 错误分类 | 视觉混淆 / 语义鸿沟 / 索引缺失 |
| 7 | 混淆矩阵 | 嵌入空间的系统性混淆模式 |
| 8 | 检索多样性 | 前 K 结果的语义覆盖度 |

## 核心发现

| 指标 | 值 | 说明 |
|------|-----|------|
| Relaxed Hit@10 | 51.5% | 同房型检索成功率 |
| Strict Hit@10 | 5.4% | 精确匹配 (1200 池) |
| 最佳文本形式 | 房型名 | MRR 是完整评论的 2.5x |

## 运行

```bash
cd Project/
python scripts/leuca/__arch_research/evaluation/test_and_analysis.py          # 全量
python scripts/leuca/__arch_research/evaluation/test_and_analysis.py --quick  # 快速
```
