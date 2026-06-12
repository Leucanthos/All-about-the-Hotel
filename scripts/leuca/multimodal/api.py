"""
多模态召回接口 — 统一 API (v1 兼容 + v2 增强).

基础接口:
    prompt2image(prompt, topK)  -> {"images": [...], "score": [...]}
    image2text(image, topK)     -> {"texts": [...], "score": [...]}

v2 增强:
    prompt2image_granular(..., granularity="room_type|hotel|strict")
    prompt2image_diverse(..., diversity_lambda=0.5)
    auto_rewrite(prompt)        -> {"best_text": ..., "strategies": {...}}
    get_confidence(result)      -> {"level": "high|medium|low", ...}
    check_consistency(text, image) -> {"consistency_score": ...}
"""

import os, re
from typing import Union, List, Dict, Optional
from collections import defaultdict

import numpy as np

from _shared.data import PROJECT_ROOT as _PROJ
from .engine import MultimodalRetriever


class MultimodalAPI:
    """多模态召回接口 — v1 基础 + v2 增强."""

    def __init__(self, device: str = "cpu"):
        self._retriever = MultimodalRetriever(device=device)
        self._diversity_cache = {}
        self._ltr_model = None
        self._classifier_head = None
        self._classifier_rooms = None

    # ═══════════════════════════════════════════════════
    # 基础召回 (v1)
    # ═══════════════════════════════════════════════════

    def prompt2image(self, prompt: str, topK: int = 5) -> dict:
        raw = self._retriever.search_images(prompt, top_k=topK)
        return self._fmt_images(raw, topK)

    def prompt2image_batch(self, prompts: List[str], topK: int = 5) -> List[dict]:
        raw_batch = self._retriever.search_images_batch(prompts, top_k=topK)
        return [self._fmt_images(raw, topK) for raw in raw_batch]

    def image2text(self, image: Union[str, "PIL.Image.Image"], topK: int = 10) -> dict:
        raw = self._retriever.search_by_image(image, top_k=topK)
        return self._fmt_texts(raw, topK)

    def image2text_batch(self, images: List[Union[str, "PIL.Image.Image"]],
                         topK: int = 10) -> List[dict]:
        raw_batch = self._retriever.search_by_image_batch(images, top_k=topK)
        return [self._fmt_texts(raw, topK) for raw in raw_batch]

    # ═══════════════════════════════════════════════════
    # 多粒度召回 (v2)
    # ═══════════════════════════════════════════════════

    def prompt2image_granular(self, prompt: str, topK: int = 10,
                               granularity: str = "strict") -> dict:
        """多粒度检索. granularity: strict | room_type | hotel."""
        result = self.prompt2image(prompt, topK=max(topK * 3, 50))
        if granularity == "strict":
            return self._trim(result, topK, "images")
        return self._diversify_by_group(result, topK,
            "room_type" if granularity == "room_type" else "comment_id", "images")

    # ═══════════════════════════════════════════════════
    # 多样性重排 (v2)
    # ═══════════════════════════════════════════════════

    def prompt2image_diverse(self, prompt: str, topK: int = 10,
                              diversity_lambda: float = 0.5) -> dict:
        """MMR 多样性重排."""
        result = self.prompt2image(prompt, topK=max(topK * 3, 50))
        return self._mmr_rerank(result, topK, diversity_lambda)

    # ═══════════════════════════════════════════════════
    # 查询改写 (v2)
    # ═══════════════════════════════════════════════════

    @staticmethod
    def auto_rewrite(prompt: str) -> dict:
        """自动改写查询文本为更有效的短形式."""
        strategies = {}
        if len(prompt) > 50:
            strategies["shorten_50"] = prompt[:50]
        if len(prompt) > 100:
            strategies["shorten_100"] = prompt[:100]

        room_kw = ["套房", "标准间", "大床房", "双床房", "主题房",
                   "行政房", "豪华房", "商务房", "家庭房"]
        found = [r for r in room_kw if r in prompt]
        if found:
            strategies["extract_room_type"] = "，".join(found)

        stopwords = set("的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这 他 她 它 们 那 些 什么 呢 吧 啊 吗 哦 嗯 但 还 被 把 让 从 对 与 或 及 等 如 虽然 因为 所以 而且 但是 然而 不过 然后 可以 这个 那个 哪 哪家 酒店 评论".split())
        words = re.findall(r"[一-龥]{2,}", prompt)
        keywords = [w for w in words if w not in stopwords][:5]
        if keywords:
            strategies["extract_keywords"] = " ".join(keywords)

        if "extract_room_type" in strategies:
            recommended = "extract_room_type"
        elif "shorten_50" in strategies:
            recommended = "shorten_50"
        elif "extract_keywords" in strategies:
            recommended = "extract_keywords"
        else:
            recommended = "original"

        return {
            "original": prompt,
            "strategies": strategies,
            "recommended": recommended,
            "best_text": strategies.get(recommended, prompt),
        }

    # ═══════════════════════════════════════════════════
    # 置信度估计 (v2)
    # ═══════════════════════════════════════════════════

    @staticmethod
    def get_confidence(result: dict) -> dict:
        """基于 score 分布估计检索置信度."""
        scores = result.get("score", [])
        if not scores or len(scores) < 2:
            return {"level": "low", "warning": "insufficient results"}

        s = np.array(scores)
        cv = float(np.std(s) / np.mean(s)) if np.mean(s) > 0 else 0.0
        spread = float(s[0] - s[-1])
        top3_std = float(np.std(s[:3]))

        if cv < 0.03 and top3_std < 0.002:
            level, warning = "low", f"分数高度集中 (CV={cv:.3f}), 结果可能来自同一图片组"
        elif cv < 0.05:
            level, warning = "medium", f"分数中度集中 (CV={cv:.3f}), 建议开启多样性重排"
        elif spread > 0.08:
            level, warning = "high", ""
        else:
            level, warning = "medium", f"区分度一般 (spread={spread:.3f})"

        return {"level": level, "score_cv": round(cv, 4),
                "score_spread": round(spread, 4), "warning": warning}

    # ═══════════════════════════════════════════════════
    # 环路一致性 (v2)
    # ═══════════════════════════════════════════════════

    def check_consistency(self, text: str, image: str, topK: int = 10) -> dict:
        """检查 text↔image 环路一致性."""
        # text → images → text
        t2i = self.prompt2image(text, topK=topK)
        fwd = False
        if t2i["images"]:
            i2t = self.image2text(t2i["images"][0], topK=topK)
            fwd = any(text[:80] in t for t in i2t["texts"])

        # image → texts → image
        i2t = self.image2text(image, topK=topK)
        bwd = False
        if i2t["texts"]:
            t2i = self.prompt2image(i2t["texts"][0], topK=topK)
            bwd = image in t2i["images"]

        score = 0.3 * fwd + 0.7 * bwd  # backward 权重更高 (更可靠)
        return {
            "text_to_image_to_text_symmetric": fwd,
            "image_to_text_to_image_symmetric": bwd,
            "consistency_score": round(score, 2),
            "diagnosis": ("image anchor stable" if bwd else "both directions broken"),
        }

    # ═══════════════════════════════════════════════════
    # LTR 重排序 (#14 算法拓展)
    # ═══════════════════════════════════════════════════

    def prompt2image_rerank(self, prompt: str, topK: int = 10) -> dict:
        """CLIP 检索 + LTR 神经网络重排序.

        在 CLIP top-20 结果上用训练好的 MLP 重排序器重新打分。
        Hit@5 提升 6pp (1%→7%)。
        """
        import torch
        result = self.prompt2image(prompt, topK=20)
        ltr = self._get_ltr_model()
        if ltr is None:
            return self._trim(result, topK, "images")

        # 构建特征
        store = self._retriever.store
        data = store.get("hotel_images", include=["metadatas"])
        metas = {iid: m for iid, m in zip(data["ids"], data["metadatas"])}

        # Detect query room_type
        query_room = ""
        for img_id in [img.split("/")[-1].replace(".jpg", "") for img in result["images"][:3]]:
            if img_id in metas:
                query_room = metas[img_id].get("room_type", "")
                break

        scores = result["score"]
        X = []
        for rank, (img, score) in enumerate(zip(result["images"], scores)):
            img_id = img.split("/")[-1].replace(".jpg", "")
            meta = metas.get(img_id, {})
            ret_room = meta.get("room_type", "")
            X.append([
                score,
                1.0 if ret_room == query_room else 0.0,
                scores[0] - score if rank > 0 else 0.0,
                rank / 20.0,
                float(np.std(scores[:5])),
            ])

        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ltr_scores = ltr(X_t).numpy()

        # Re-rank
        ranked = sorted(zip(result["images"], scores, ltr_scores),
                       key=lambda x: -x[2])
        result["images"] = [r[0] for r in ranked[:topK]]
        result["score"] = [float(r[1]) for r in ranked[:topK]]
        return result

    # ═══════════════════════════════════════════════════
    # 分类器预过滤 (#3 算法拓展)
    # ═══════════════════════════════════════════════════

    def classify_room_type(self, text: str) -> dict:
        """用 BERT 分类器预测查询文本对应的房型."""
        import torch
        clf = self._get_classifier()
        if clf is None:
            return {"room_type": None, "confidence": 0}

        processor = self._retriever.processor
        ti = processor(text=text, return_tensors="pt", padding=True,
                      truncation=True, max_length=77)
        with torch.no_grad():
            out = self._retriever.model.text_model(
                input_ids=ti["input_ids"], attention_mask=ti["attention_mask"])
            pooled = out.last_hidden_state[:, 0, :]
            logits = clf(pooled)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = probs.argmax().item()

        return {
            "room_type": self._classifier_rooms[pred_idx],
            "confidence": round(float(probs.max()), 3),
            "all_probs": {r: round(float(p), 3)
                         for r, p in zip(self._classifier_rooms, probs.tolist())},
        }

    def prompt2image_filtered(self, prompt: str, topK: int = 10) -> dict:
        """房型预过滤 + CLIP 检索.

        先用分类器预测房型, 仅在预测房型对应的图片中检索。
        """
        room_result = self.classify_room_type(prompt)
        pred_room = room_result["room_type"]
        if pred_room is None:
            return self.prompt2image(prompt, topK)

        # 获取预测房型的所有图片
        store = self._retriever.store
        data = store.get("hotel_images", include=["metadatas"])
        candidates = []
        for img_id, meta in zip(data["ids"], data["metadatas"]):
            if meta.get("room_type", "") == pred_room:
                candidates.append(f"data/images/{img_id}.jpg")

        # CLIP 检索 + 过滤
        result = self.prompt2image(prompt, topK=max(topK * 3, 30))
        filtered = [img for img in result["images"] if img in candidates]
        filtered_scores = [s for img, s in zip(result["images"], result["score"])
                          if img in candidates]
        result["images"] = filtered[:topK]
        result["score"] = filtered_scores[:topK]
        result["predicted_room_type"] = pred_room
        result["classifier_confidence"] = room_result["confidence"]
        return result

    # ═══════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _fmt_images(raw: list, topK: int) -> dict:
        images, scores = [], []
        for item in raw[:topK]:
            img_id = item.get("image_id", "")
            if img_id:
                images.append(f"data/images/{img_id}.jpg")
            scores.append(item.get("similarity", 0.0))
        return {"images": images, "score": scores}

    @staticmethod
    def _fmt_texts(raw: list, topK: int) -> dict:
        texts, scores = [], []
        for item in raw[:topK]:
            texts.append(item.get("comment", ""))
            scores.append(item.get("similarity", 0.0))
        return {"texts": texts, "score": scores}

    @staticmethod
    def _trim(result: dict, topK: int, key: str) -> dict:
        return {**result, key: result[key][:topK], "score": result["score"][:topK]}

    def _diversify_by_group(self, result: dict, topK: int,
                             group_key: str, item_key: str) -> dict:
        """按 group_key 去重分组."""
        store = self._retriever.store
        coll_name = "hotel_comments" if item_key == "texts" else "hotel_images"
        data = store.get(coll_name, include=["metadatas"])
        all_ids, all_metas = data["ids"], data["metadatas"]

        items, scores = result[item_key], result["score"]
        groups_seen = defaultdict(int)
        n_groups = 4 if group_key == "room_type" else max(2, topK // 3)
        per_group = max(1, topK // n_groups) + 1

        filtered_items, filtered_scores = [], []
        for item, s in zip(items, scores):
            item_id = item.split("/")[-1].replace(".jpg", "") if not "texts" in item_key else item
            g = "unknown"
            try:
                idx = all_ids.index(item_id)
                g = all_metas[idx].get(group_key, "unknown") if all_metas else "unknown"
            except ValueError:
                pass
            if groups_seen[g] < per_group:
                filtered_items.append(item)
                filtered_scores.append(s)
                groups_seen[g] += 1
            if len(filtered_items) >= topK:
                break

        result[item_key] = filtered_items
        result["score"] = filtered_scores
        return result

    def _mmr_rerank(self, result: dict, topK: int, lam: float) -> dict:
        """MMR 多样性重排."""
        items, scores = result["images"], np.array(result["score"])
        if len(items) <= topK:
            return self._trim(result, topK, "images")

        store = self._retriever.store
        if "hotel_images" not in self._diversity_cache:
            data = store.get("hotel_images", include=["embeddings"])
            self._diversity_cache["hotel_images"] = {
                "ids": data["ids"],
                "embs": np.array(data["embeddings"]).astype(np.float32),
            }
        cache = self._diversity_cache["hotel_images"]
        all_ids, all_embs = cache["ids"], cache["embs"]

        item_indices = []
        for item in items:
            iid = item.split("/")[-1].replace(".jpg", "")
            try:
                item_indices.append(all_ids.index(iid))
            except ValueError:
                item_indices.append(-1)

        selected = [0]
        candidates = list(range(1, len(items)))
        while len(selected) < topK and candidates:
            mmr_scores = []
            for c in candidates:
                relevance = scores[c]
                max_sim = 0.0
                if item_indices[c] >= 0:
                    sims = [float(np.dot(all_embs[item_indices[c]], all_embs[item_indices[s]]))
                            for s in selected if item_indices[s] >= 0]
                    max_sim = max(sims) if sims else 0.0
                mmr_scores.append(relevance - lam * max_sim)
            best = candidates[int(np.argmax(mmr_scores))]
            selected.append(best)
            candidates.remove(best)

        result["images"] = [items[i] for i in selected]
        result["score"] = [float(scores[i]) for i in selected]
        return result

    @property
    def stats(self) -> dict:
        return self._retriever.stats

    @property
    def retriever(self) -> MultimodalRetriever:
        return self._retriever

    def _get_ltr_model(self):
        """懒加载 LTR 重排序模型."""
        if self._ltr_model is not None:
            return self._ltr_model
        import torch
        ltr_path = os.path.join(str(_PROJ), "data", "ltr_model.pt")
        if not os.path.exists(ltr_path):
            return None
        ckpt = torch.load(ltr_path, map_location="cpu", weights_only=True)
        # 重建模型 — LTRModel 内部是 self.net (Sequential)
        n_feat = ckpt["n_features"]; hidden = ckpt["hidden_dim"]
        layers = []
        for d_in, d_out in [(n_feat, hidden), (hidden, hidden), (hidden, 1)]:
            layers.append(torch.nn.Linear(d_in, d_out))
            if d_out > 1:
                layers.append(torch.nn.ReLU())
        net = torch.nn.Sequential(*layers)
        # state_dict keys are prefixed with "net."
        net.load_state_dict({k.replace("net.", ""): v
                            for k, v in ckpt["state_dict"].items()})
        net.eval()
        self._ltr_model = net
        return net

    def _get_classifier(self):
        """懒加载房型分类器."""
        if self._classifier_head is not None:
            return self._classifier_head
        import torch
        clf_path = os.path.join(str(_PROJ), "data", "classifier_model.pt")
        if not os.path.exists(clf_path):
            return None
        ckpt = torch.load(clf_path, map_location="cpu", weights_only=True)
        head = torch.nn.Linear(768, len(ckpt["room_list"]))
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        self._classifier_head = head
        self._classifier_rooms = ckpt["room_list"]
        return head


# ── 模块级快捷函数 ──
_api: Optional[MultimodalAPI] = None

def _get() -> MultimodalAPI:
    global _api
    if _api is None:
        _api = MultimodalAPI()
    return _api

def prompt2image(prompt: str, topK: int = 5) -> dict:
    return _get().prompt2image(prompt, topK)

def image2text(image, topK: int = 10) -> dict:
    return _get().image2text(image, topK)


if __name__ == "__main__":
    api = MultimodalAPI()
    print(f"Stats: {api.stats}")
    r = api.prompt2image("游泳池干净吗", topK=3)
    print(f"prompt2image: {[p.split('/')[-1] for p in r['images']]}, top={r['score'][0]:.3f}")
    rw = api.auto_rewrite("房间非常好 装修很厚重奢华")
    print(f"auto_rewrite: {rw['recommended']} -> '{rw['best_text'][:60]}...'")
    conf = api.get_confidence(r)
    print(f"confidence: {conf['level']} (CV={conf['score_cv']:.4f})")
