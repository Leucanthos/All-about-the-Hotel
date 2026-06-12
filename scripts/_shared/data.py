from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA = PROJECT_ROOT / 'data'

# 原始数据
RAW_DIR = DATA / 'raw'
COMMENTS_PATH = RAW_DIR / 'hotel_reviews_table.parquet'
COMMENTS_FULL_PATH = DATA / 'hotel_reviews_full.parquet'  # 多酒店数据集 (694 hotels, 414k reviews)
SUMMARIES_PATH = RAW_DIR / 'category_summaries.json'
REVERSE_QUERIES_PATH = RAW_DIR / 'reverse_queries.csv'
SFT_DIR = DATA / 'sft'

# 资源
IMAGES_DIR = DATA / 'images'
MODELS_DIR = DATA / 'models'
VECTORS_DIR = DATA / 'vectors'
SPLIT_DIR = DATA / 'split'
