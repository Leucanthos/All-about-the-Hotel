# LTR — Learning to Rank 重排序实验

**方法**: 在 CLIP 检索 top-20 结果上训练 MLP 重排序器

## 特征设计 (R0-R6 迭代)

| 轮次 | 新增特征 | 说明 |
|------|----------|------|
| R0 | CLIP cosine | 基线: 仅用 CLIP 相似度 |
| R1 | + same_room | 是否同一房型 |
| R2 | + score_drop | 与 top-1 的分数差 |
| R3 | + norm_rank | 归一化排名 |
| R4 | + cross_modal | 跨模态特征 (I2T + T2I) |
| R5 | + hard_neg | 难负样本加权损失 |
| R6 | + ensemble | 多模型集成 |

## 核心发现

- 最佳结果: R3 (CLIP+LTR) Hit@5 提升 ~6pp (1%→7%)
- R4-R6 无进一步提升，说明 CLIP 特征本身是瓶颈
- 分类器预过滤是更有效的算法拓展方向

## 运行

```bash
cd Project/
python scripts/leuca/__arch_research/ltr/experiments.py --task ltr --round 4
python scripts/leuca/__arch_research/ltr/experiments.py --task ltr --report
```
