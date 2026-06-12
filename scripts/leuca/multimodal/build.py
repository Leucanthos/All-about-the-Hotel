"""
大规模索引构建 — 下载图片并重建 NumPy 向量索引.

用法:
    python build_index.py              # 默认 1200/2000
    python build_index.py --images 300 --texts 500
"""

import sys, os, json, time, warnings
from io import BytesIO
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from PIL import Image

warnings.filterwarnings("ignore")

_this_dir = os.path.dirname(os.path.abspath(__file__))
_deliverable_dir = os.path.dirname(_this_dir)
_project_root = os.path.dirname(os.path.dirname(_deliverable_dir))
sys.path.insert(0, os.path.join(_project_root, "scripts"))
sys.path.insert(0, _this_dir)  # peer imports (numpy_vector_store)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HotelReview/1.0)"}
IMAGE_DIR = os.path.join(_project_root, "data", "images")
VECTOR_DIR = os.path.join(_project_root, "data", "vectors")


def download_images(df: pd.DataFrame, target_count: int = 500) -> list:
    """下载图片，每唯一 _id 取首张。返回 [(img_path, row_data), ...]."""
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 按 _id 去重，取每组第一条
    unique_reviews = df.drop_duplicates(subset="_id")

    # 跨房型均匀采样
    room_groups = defaultdict(list)
    for idx, row in unique_reviews.iterrows():
        room_groups[str(row["fuzzy_room_type"])].append((idx, row))

    per_room = max(1, target_count // len(room_groups))
    sampled = []
    for room, items in room_groups.items():
        sampled.extend(items[:per_room])
    # 补足
    if len(sampled) < target_count:
        remaining = []
        for room, items in room_groups.items():
            remaining.extend(items[per_room:])
        sampled.extend(remaining[:target_count - len(sampled)])

    print(f"Target: {target_count} images from {len(unique_reviews)} unique reviews")
    print(f"Sampled: {len(sampled)} across {len(room_groups)} room types")
    for room, items in room_groups.items():
        taken = sum(1 for idx, _ in sampled if str(_.get("fuzzy_room_type", "")) == room)
        print(f"  {room}: {taken}/{len(items)}")

    # 下载
    results = []
    downloaded, skipped, failed = 0, 0, 0
    for idx, row in sampled:
        img_name = f"img_{idx}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_name)

        if os.path.exists(img_path) and os.path.getsize(img_path) > 100:
            skipped += 1
            results.append((img_path, row, idx))
            continue

        try:
            urls = json.loads(row["images"])
            if not urls:
                failed += 1
                continue
            resp = requests.get(urls[0], timeout=15, headers=HEADERS)
            if resp.status_code != 200:
                failed += 1
                continue
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(img_path, quality=85)
            downloaded += 1
            results.append((img_path, row, idx))
            if downloaded % 50 == 0:
                print(f"  Downloaded: {downloaded}, skipped: {skipped}, failed: {failed}")
        except Exception:
            failed += 1
            continue

    print(f"Done: {len(results)} images (downloaded={downloaded}, skipped={skipped}, failed={failed})")
    return results


def build_indices(image_list: list, df: pd.DataFrame, text_limit: int = 1500):
    """构建/重建 Chroma 向量索引."""
    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    # ── 加载模型 ──
    print("\nLoading Chinese-CLIP...")
    device = "cpu"
    model = ChineseCLIPModel.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True
    ).eval().to(device)
    processor = ChineseCLIPProcessor.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True
    )

    # ── NumPy 向量存储 ──
    from numpy_vector_store import NumpyVectorStore
    store_dir = os.path.join(_project_root, "data", "vectors_np")
    os.makedirs(store_dir, exist_ok=True)
    store = NumpyVectorStore(store_dir)

    # ── 文本索引 ──
    print(f"\nIndexing texts (limit={text_limit})...")
    df_unique = df.drop_duplicates(subset="comment").head(text_limit)

    txt_ids, txt_embs_all, txt_meta_all, txt_docs_all = [], [], [], []
    txt_batch = 32
    for batch_start in range(0, len(df_unique), txt_batch):
        batch_rows = df_unique.iloc[batch_start:batch_start + txt_batch]
        texts = [str(r["comment"])[:300] for _, r in batch_rows.iterrows()]
        ids = [str(r["_id"]) for _, r in batch_rows.iterrows()]
        metas = [{
            "comment_id": str(r["_id"]), "content": t,
            "score": float(r["score"]),
            "publish_date": str(r.get("publish_date", "")),
            "room_type": str(r.get("fuzzy_room_type", "")),
        } for (_, r), t in zip(batch_rows.iterrows(), texts)]

        inputs = processor(text=texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=77).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs = feats.cpu().numpy().astype(np.float32)

        txt_ids.extend(ids)
        txt_embs_all.append(embs)
        txt_meta_all.extend(metas)
        txt_docs_all.extend(texts)
        if (batch_start // txt_batch) % 20 == 0:
            print(f"  Texts: {min(batch_start + txt_batch, len(df_unique))}/{len(df_unique)}")

    store.create("hotel_comments", txt_ids,
                 np.concatenate(txt_embs_all, axis=0),
                 txt_meta_all, txt_docs_all)
    print(f"  Text index: {store.count('hotel_comments')} vectors")

    # ── 图片索引 ──
    print(f"\nIndexing {len(image_list)} images...")
    img_ids, img_embs_all, img_meta_all, img_docs_all = [], [], [], []
    batch_size = 16
    for batch_start in range(0, len(image_list), batch_size):
        batch = image_list[batch_start:batch_start + batch_size]
        imgs = [Image.open(p).convert("RGB") for p, _, _ in batch]
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs = feats.cpu().numpy().astype(np.float32)

        for (_, row, idx), emb in zip(batch, embs):
            img_id = f"img_{idx}"
            img_ids.append(img_id)
            img_embs_all.append(emb)
            meta = {
                "image_id": img_id,
                "comment_id": str(row.get("_id", idx)),
                "comment": str(row["comment"])[:200],
                "score": float(row["score"]),
                "room_type": str(row.get("fuzzy_room_type", "")),
            }
            img_meta_all.append(meta)
            img_docs_all.append(str(row["comment"])[:200])

        if (batch_start // batch_size) % 10 == 0:
            print(f"  Images: {min(batch_start + batch_size, len(image_list))}/{len(image_list)}")

    store.create("hotel_images", img_ids,
                 np.array(img_embs_all, dtype=np.float32),
                 img_meta_all, img_docs_all)
    print(f"  Image index: {store.count('hotel_images')} vectors")
    print("\nBuild complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=1200)
    parser.add_argument("--texts", type=int, default=2000)
    args = parser.parse_args()

    start = time.time()
    print("=" * 60)
    print(f"Scaling index: {args.images} images + {args.texts} texts")
    print("=" * 60)

    df = pd.read_parquet(os.path.join(_project_root, "data", "hotel_reviews_table.parquet"))
    print(f"Data: {len(df)} rows\n")

    # 下载
    img_list = download_images(df, target_count=args.images)

    # 构建索引
    build_indices(img_list, df, text_limit=args.texts)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s")
