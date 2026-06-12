"""共享数据路径 — leuca / yuhao 统一从此读取.

目录结构:
  data/
  ├── raw/                 原始数据
  │   ├── hotel_reviews_table.parquet
  │   └── category_summaries.json
  ├── images/              酒店图片 (1233 张)
  ├── models/              训练好的模型权重
  │   ├── ltr_model.pt
  │   └── classifier_model.pt
  ├── vectors/             预计算向量索引
  │   └── hotel_{images,comments}_*.npy
  └── split/               Train/Test 划分
"""

from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA = PROJECT_ROOT / "data"

# 原始数据
RAW_DIR = DATA / "raw"
COMMENTS_PATH = RAW_DIR / "hotel_reviews_table.parquet"
SUMMARIES_PATH = RAW_DIR / "category_summaries.json"
REVERSE_QUERIES_PATH = RAW_DIR / "reverse_queries.csv"
SFT_DIR = DATA / "sft"

# 资源
IMAGES_DIR = DATA / "images"
MODELS_DIR = DATA / "models"
VECTORS_DIR = DATA / "vectors"
SPLIT_DIR = DATA / "split"
