"""Chinese-CLIP 多模态检索接口."""
import os, json, warnings
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image

warnings.filterwarnings("ignore")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HotelReview/1.0)"}


class MultimodalRetriever:
    """中文多模态检索引擎.

    Usage:
        mr = MultimodalRetriever()
        mr.build(data_dir="data")
        results = mr.search("早餐怎么样")
        img_results = mr.search_images("游泳池干净吗")
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._processor = None
        self._chroma = None
        self._comment_df = None

    # ── lazy load ──
    @property
    def model(self):
        if self._model is None:
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            print("Loading Chinese-CLIP...")
            self._model = ChineseCLIPModel.from_pretrained(
                "OFA-Sys/chinese-clip-vit-base-patch16"
            ).eval().to(self.device)
            self._processor = ChineseCLIPProcessor.from_pretrained(
                "OFA-Sys/chinese-clip-vit-base-patch16"
            )
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            _ = self.model  # trigger load
        return self._processor

    @property
    def chroma(self):
        if self._chroma is None:
            import chromadb
            self._chroma = chromadb.PersistentClient(path="./data/chromadb_cn")
        return self._chroma

    @property
    def comments(self) -> pd.DataFrame:
        if self._comment_df is None:
            path = "filtered_comments.parquet"
            if os.path.exists(path):
                self._comment_df = pd.read_parquet(path)
            else:
                self._comment_df = pd.read_csv("filtered_comments.csv")
        return self._comment_df

    # ── embedding helpers ──
    def _embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        with torch.no_grad():
            feat = self.model.get_text_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze().cpu().numpy().astype(np.float32)

    def _load_image(self, source: Union[str, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if source.startswith(("http://", "https://")):
            resp = requests.get(source, timeout=10, headers=HEADERS)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        return Image.open(source).convert("RGB")

    # ── build index ──
    def build(
        self,
        data_dir: str = "data",
        image_limit: int = 50,
        text_limit: int = 300,
    ):
        """构建/重建多模态索引."""
        os.makedirs(data_dir, exist_ok=True)
        image_dir = os.path.join(data_dir, "cn_images")
        os.makedirs(image_dir, exist_ok=True)
        df = self.comments

        # ── image index ──
        print(f"Indexing images (limit={image_limit})...")
        img_embs, img_meta, img_ids = [], [], []

        for idx, row in df.head(image_limit).iterrows():
            try:
                urls = json.loads(row["images"])
                if not urls:
                    continue
                resp = requests.get(urls[0], timeout=10, headers=HEADERS)
                if resp.status_code != 200:
                    continue
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img_id = f"img_{idx}"
                img.save(os.path.join(image_dir, f"{img_id}.jpg"), quality=85)

                emb = self._embed_image(img)
                img_embs.append(emb.tolist())
                img_meta.append({
                    "image_id": img_id,
                    "comment_id": str(row.get("_id", idx)),
                    "comment": str(row["comment"])[:200],
                    "score": float(row["score"]),
                    "room_type": str(row.get("fuzzy_room_type", "")),
                })
                img_ids.append(img_id)
            except Exception:
                continue

        img_coll = self.chroma.get_or_create_collection(
            "hotel_images", metadata={"hnsw:space": "cosine"}
        )
        if img_embs:
            img_coll.upsert(ids=img_ids, embeddings=img_embs, metadatas=img_meta,
                           documents=[m["comment"] for m in img_meta])
        print(f"  Images: {len(img_embs)}")

        # ── text index ──
        print(f"Indexing texts (limit={text_limit})...")
        txt_embs, txt_meta, txt_ids = [], [], []

        for idx, row in df.head(text_limit).iterrows():
            comment = str(row["comment"])[:300]
            if not comment.strip():
                continue
            emb = self._embed_text([comment])[0]
            txt_embs.append(emb.tolist())
            txt_meta.append({
                "comment_id": str(row.get("_id", idx)),
                "content": comment,
                "score": float(row["score"]),
                "publish_date": str(row.get("publish_date", "")),
                "room_type": str(row.get("fuzzy_room_type", "")),
            })
            txt_ids.append(str(row.get("_id", idx)))

        txt_coll = self.chroma.get_or_create_collection(
            "hotel_comments", metadata={"hnsw:space": "cosine"}
        )
        if txt_embs:
            txt_coll.upsert(ids=txt_ids, embeddings=txt_embs, metadatas=txt_meta,
                           documents=[m["content"] for m in txt_meta])
        print(f"  Texts: {len(txt_embs)}")

        print("Build complete.")

    # ── search APIs ──
    def _format_results(self, results, top_k: int) -> list[dict]:
        out = []
        if not results["ids"] or not results["ids"][0]:
            return out
        for i in range(min(top_k, len(results["ids"][0]))):
            sim = max(0.0, 1.0 - results["distances"][0][i])
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            out.append({
                "rank": i + 1,
                "similarity": round(sim, 4),
                "comment": meta.get("comment") or meta.get("content", "")[:200],
                "score": meta.get("score"),
                "room_type": meta.get("room_type"),
                "image_id": meta.get("image_id"),
            })
        return out

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """文本检索 — 用中文查询搜评论文本."""
        q_emb = self._embed_text([query])[0].tolist()
        coll = self.chroma.get_collection("hotel_comments")
        results = coll.query(query_embeddings=[q_emb], n_results=top_k)
        return self._format_results(results, top_k)

    def search_images(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """文本 → 图片 — 用中文查询搜相关酒店图片."""
        q_emb = self._embed_text([query])[0].tolist()
        coll = self.chroma.get_collection("hotel_images")
        results = coll.query(query_embeddings=[q_emb], n_results=top_k)
        return self._format_results(results, top_k)

    def search_by_image(
        self,
        image: Union[str, Image.Image],
        top_k: int = 10,
    ) -> list[dict]:
        """图片 → 文本 — 上传图片搜相关评论."""
        img = self._load_image(image)
        emb = self._embed_image(img).tolist()
        coll = self.chroma.get_collection("hotel_comments")
        results = coll.query(query_embeddings=[emb], n_results=top_k)
        return self._format_results(results, top_k)

    @property
    def stats(self) -> dict:
        try:
            img_coll = self.chroma.get_collection("hotel_images")
            txt_coll = self.chroma.get_collection("hotel_comments")
            return {
                "images": img_coll.count(),
                "texts": txt_coll.count(),
                "comments_total": len(self.comments),
            }
        except Exception:
            return {"images": 0, "texts": 0, "comments_total": 0}


# ── quick test ──
if __name__ == "__main__":
    mr = MultimodalRetriever()
    print(f"Stats: {mr.stats}")

    # Text search
    print("\n=== search('早餐怎么样') ===")
    for r in mr.search("早餐怎么样", top_k=3):
        print(f"  [{r['similarity']:.3f}] {r['comment'][:80]}...")

    # Cross-modal
    print("\n=== search_images('游泳池') ===")
    for r in mr.search_images("游泳池", top_k=3):
        print(f"  [{r['similarity']:.3f}] {r['comment'][:80]}...")

    print("\nDone.")
