# Fine-tuning — Chinese-CLIP 对比微调实验

**方法**: 对比学习微调 Chinese-CLIP 投影头 (786K / 188M params)

## 实验矩阵

| 实验 | 命令 | 说明 |
|------|------|------|
| baseline | `--experiment baseline` | 仅训练 text_projection + visual_projection |
| unfreeze_top | `--experiment unfreeze_top` | 解冻顶层 encoder layer |
| augmentation | `--experiment augmentation` | 图像裁剪 + 文本随机截断 |
| temp_smooth | `--experiment temp_smooth` | 可学习温度 + label smoothing 0.1 |
| long_text | `--experiment long_text` | max_length=200 (默认77) |
| i2t_weight | `--experiment i2t_weight` | I2T loss 权重 3x |
| hard_neg | `--experiment hard_neg` | 难负样本加权 2x |

## 核心发现

- 仅微调投影头: T2I Hit@1 从 0.06% → 0.38% (train)，但零测试泛化
- 解冻顶层: train 100% 但完全过拟合，test=0
- 数据增强: T2I Hit@1 反而下降到 0.18%
- **结论**: Chinese-CLIP 的预训练投影头已经接近最优，小规模 fine-tune 难以超越

## 运行

```bash
cd Project/
python scripts/leuca/__arch_research/finetune/finetune.py --experiment baseline --epochs 10
python scripts/leuca/__arch_research/finetune/finetune.py --rebuild-only --experiment baseline
```
