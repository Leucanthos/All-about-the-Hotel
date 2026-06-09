"""Chinese-CLIP 多模态检索 — NumPy 向量存储后端.

    from engine.multimodal import MultimodalRetriever

    mr = MultimodalRetriever()
    mr.build()                              # 首次运行：构建索引

    mr.search_images("游泳池干净吗")          # 文本 → 图片
    mr.search_images_batch(["早餐","泳池"])   #   批量版（一次 forward）
    mr.search_by_image("img.jpg")            # 图片 → 文本
    mr.search_by_image_batch(["a.jpg","b.jpg"])  #   批量版
    mr.stats                                 # {'images': 500, 'texts': 2000, ...}
"""
import sys, os, warnings, threading
from io import BytesIO
from typing import Union

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image

warnings.filterwarnings("ignore")

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HotelReview/1.0)"}

# vibecoding/src 通过 sys.path 加载
_vibecoding_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "vibecoding", "src")
if _vibecoding_src not in sys.path:
    sys.path.insert(0, _vibecoding_src)


class MultimodalRetriever:
    """中文多模态检索引擎 — NumPy 后端，线程安全."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._processor = None
        self._store = None
        self._comment_df = None
        self._lock = threading.Lock()

    # ── lazy load (thread-safe) ──
    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._load_model()
        return self._model

    def _load_model(self):
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        print("Loading Chinese-CLIP...")
        self._model = ChineseCLIPModel.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True
        ).eval().to(self.device)
        self._processor = ChineseCLIPProcessor.from_pretrained(
            "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True
        )
        # 自动加载微调投影头 (如果存在)
        ft_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "finetuned")
        ft_weights = os.path.join(ft_dir, "projection_heads.pt")
        if os.path.exists(ft_weights):
            state = torch.load(ft_weights, map_location=self.device, weights_only=True)
            self._model.text_projection.load_state_dict(state["text_projection"])
            self._model.visual_projection.load_state_dict(state["visual_projection"])
            print("  [Fine-tuned projection heads loaded]")

        # 尝试加载 OpenVINO GPU 加速
        self._ov_text = None
        self._ov_vision = None
        self._ov_text_names = None
        self._ov_vision_names = None
        self._try_load_openvino()

    @property
    def processor(self):
        if self._processor is None:
            _ = self.model
        return self._processor

    @property
    def store(self):
        if self._store is None:
            with self._lock:
                if self._store is None:
                    from numpy_vector_store import NumpyVectorStore
                    self._store = NumpyVectorStore()
        return self._store

    @property
    def comments(self) -> pd.DataFrame:
        if self._comment_df is None:
            self._comment_df = pd.read_parquet("data/hotel_reviews_table.parquet")
        return self._comment_df

    def _try_load_openvino(self):
        """尝试加载 OpenVINO GPU 加速模型.

        注意: 如果存在微调投影头，跳过 OpenVINO (IR 模型是原始权重, 与微调不兼容).
        """
        ft_weights = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "finetuned", "projection_heads.pt")
        if os.path.exists(ft_weights):
            print("  [Fine-tuned weights detected, using PyTorch (skip OpenVINO)]")
            return

        try:
            import openvino as ov
            ov_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "data", "openvino")
            text_xml = os.path.join(ov_dir, "text_encoder.xml")
            vision_xml = os.path.join(ov_dir, "vision_encoder.xml")
            if not os.path.exists(text_xml) or not os.path.exists(vision_xml):
                return

            core = ov.Core()
            if "GPU" not in core.available_devices:
                return

            text_ov = core.read_model(text_xml)
            vision_ov = core.read_model(vision_xml)
            self._ov_text = core.compile_model(text_ov, "GPU")
            self._ov_vision = core.compile_model(vision_ov, "GPU")
            self._ov_text_names = [i.get_any_name() for i in text_ov.inputs]
            self._ov_vision_names = [i.get_any_name() for i in vision_ov.inputs]
            print(f"  [OpenVINO GPU loaded: text={self._ov_text_names}, vision={self._ov_vision_names}]")
        except Exception as e:
            pass  # 静默回退到 PyTorch CPU

    # ── embedding ──
    def _embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        # OpenVINO GPU 路径
        if self._ov_text is not None:
            ov_in = {self._ov_text_names[0]: inputs["input_ids"].numpy(),
                     self._ov_text_names[1]: inputs["attention_mask"].numpy()}
            result = self._ov_text(ov_in)
            feat = result[0]
            feat = feat / np.linalg.norm(feat, axis=-1, keepdims=True)
            return feat.astype(np.float32)

        # PyTorch 路径 — 手动 forward 避免 get_text_features 兼容问题
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            t_out = self.model.text_model(**inputs)
            feat = self.model.text_projection(t_out.last_hidden_state[:, 0, :])
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        # OpenVINO GPU 路径
        if self._ov_vision is not None:
            ov_in = {self._ov_vision_names[0]: inputs["pixel_values"].numpy()}
            result = self._ov_vision(ov_in)
            feat = result[0]
            feat = feat / np.linalg.norm(feat, axis=-1, keepdims=True)
            return feat.squeeze().astype(np.float32)

        # PyTorch 路径 — 手动 forward
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            i_out = self.model.vision_model(**inputs)
            feat = self.model.visual_projection(i_out.last_hidden_state[:, 0, :])
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

    # ── search ──
    def _dump(self, results, top_k: int) -> list[dict]:
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

    def search_images(self, query: str, top_k: int = 5) -> list[dict]:
        """文本 → 图片."""
        return self.search_images_batch([query], top_k)[0]

    def search_images_batch(self, queries: list[str], top_k: int = 5) -> list[list[dict]]:
        """文本 → 图片 (批量)."""
        embs = self._embed_text(queries)
        return [
            self._dump(self.store.query("hotel_images",
                        query_embeddings=[e.tolist()], n_results=top_k), top_k)
            for e in embs
        ]

    def search_by_image(self, image: Union[str, Image.Image], top_k: int = 10) -> list[dict]:
        """图片 → 文本."""
        return self.search_by_image_batch([image], top_k)[0]

    def search_by_image_batch(self, images: list[Union[str, Image.Image]], top_k: int = 10) -> list[list[dict]]:
        """图片 → 文本 (批量)."""
        imgs = [self._load_image(img) for img in images]
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            i_out = self.model.vision_model(**inputs)
            feats = self.model.visual_projection(i_out.last_hidden_state[:, 0, :])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs = feats.cpu().numpy().astype(np.float32)
        return [
            self._dump(self.store.query("hotel_comments",
                        query_embeddings=[e.tolist()], n_results=top_k), top_k)
            for e in embs
        ]

    @property
    def stats(self) -> dict:
        try:
            return {
                "images": self.store.count("hotel_images"),
                "texts": self.store.count("hotel_comments"),
                "comments_total": len(self.comments),
            }
        except Exception:
            return {"images": 0, "texts": 0, "comments_total": 0}


if __name__ == "__main__":
    mr = MultimodalRetriever()
    print(f"Stats: {mr.stats}")

    print("\n=== search_images('游泳池') ===")
    for r in mr.search_images("游泳池", top_k=3):
        print(f"  [{r['similarity']:.3f}] {r['comment'][:80]}...")

    print("\nDone.")
