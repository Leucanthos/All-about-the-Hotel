"""索引构建: 下载图片 + NumPy 向量索引."""
import sys, os

def build(project_root: str, n_images: int = 500, n_texts: int = 1000):
    print(f"\nBuilding index ({n_images} images, {n_texts} texts)...")
    sys.path.insert(0, os.path.join(project_root, "scripts"))
    from leuca.multimodal.build import download_images, build_indices
    import pandas as pd

    df = pd.read_parquet(os.path.join(project_root, "data", "raw", "hotel_reviews_table.parquet"))
    img_list = download_images(df, target_count=n_images)
    build_indices(img_list, df, text_limit=n_texts)
    print("[OK] Index built")
