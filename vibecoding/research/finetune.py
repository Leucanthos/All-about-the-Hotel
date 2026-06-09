"""
对比微调 Chinese-CLIP — 可配置实验框架.

用法:
    python finetune.py                                  # 基线
    python finetune.py --experiment unfreeze_top        # 实验1: 解冻顶层
    python finetune.py --experiment augmentation        # 实验2: 数据增强
    python finetune.py --experiment temp_smooth         # 实验3: 温度+平滑
    python finetune.py --experiment long_text           # 实验4: 长文本
    python finetune.py --experiment i2t_weight          # 实验5: I2T加权
    python finetune.py --experiment hard_neg            # 实验6: 难负样本
    python finetune.py --rebuild-only                   # 仅重建索引
"""
import sys, os, json, time, warnings, argparse, random
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

_this_dir = os.path.dirname(os.path.abspath(__file__))
_deliverable_dir = os.path.dirname(_this_dir)
_project_root = os.path.dirname(_deliverable_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_deliverable_dir, "src"))

from numpy_vector_store import NumpyVectorStore

DEVICE = "xpu" if hasattr(__import__("torch"), "xpu") and __import__("torch").xpu.is_available() else "cpu"


# ═══════════════════════════════════════════════════
# Experiment Config
# ═══════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    name: str = "baseline"
    epochs: int = 10
    batch_size: int = 16
    lr: float = 5e-5
    max_length: int = 77
    temperature: float = 0.07
    label_smoothing: float = 0.0
    learnable_temp: bool = False
    i2t_weight: float = 1.0  # >1 means I2T loss weighted more
    trainable_patterns: List[str] = field(default_factory=lambda: [
        "text_projection", "visual_projection"
    ])
    use_augmentation: bool = False
    hard_neg_boost: float = 0.0  # 0 = off, >0 = boost hard negatives
    backbone_in_eval: bool = True  # keep backbone in eval mode vs train mode


# ── 实验预设 ──

PRESETS = {
    "baseline": ExperimentConfig(
        name="baseline",
        trainable_patterns=["text_projection", "visual_projection"],
        backbone_in_eval=True,
    ),
    "unfreeze_top": ExperimentConfig(
        name="unfreeze_top",
        trainable_patterns=[
            "text_projection", "visual_projection",
            "text_model.encoder.layer.11",
            "vision_model.encoder.layers.11",
        ],
        backbone_in_eval=False,
    ),
    "augmentation": ExperimentConfig(
        name="augmentation",
        use_augmentation=True,
    ),
    "temp_smooth": ExperimentConfig(
        name="temp_smooth",
        learnable_temp=True,
        label_smoothing=0.1,
    ),
    "long_text": ExperimentConfig(
        name="long_text",
        max_length=200,
    ),
    "i2t_weight": ExperimentConfig(
        name="i2t_weight",
        i2t_weight=3.0,
    ),
    "hard_neg": ExperimentConfig(
        name="hard_neg",
        hard_neg_boost=2.0,
    ),
    # 组合实验
    "combo_aug_unfreeze": ExperimentConfig(
        name="combo_aug_unfreeze",
        trainable_patterns=[
            "text_projection", "visual_projection",
            "text_model.encoder.layer.11",
            "vision_model.encoder.layers.11",
        ],
        backbone_in_eval=False,
        use_augmentation=True,
    ),
}


# ═══════════════════════════════════════════════════
# Data Augmentation
# ═══════════════════════════════════════════════════

try:
    from torchvision import transforms as T
    _aug_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    ])
    _has_torchvision = True
except ImportError:
    _aug_transform = None
    _has_torchvision = False


# ═══════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════

class PairDataset:
    def __init__(self, processor, df, store, cfg: ExperimentConfig, max_pairs=800):
        self.processor = processor
        self.cfg = cfg
        print("  Loading image metadata...")
        coll_name = "hotel_images_train" if store.exists("hotel_images_train") else "hotel_images"
        data = store.get(coll_name, include=["metadatas"])
        print(f"  Building pairs from {len(data['ids'])} images ({coll_name})...")
        self.pairs = []
        for img_id, meta in zip(data["ids"], data["metadatas"]):
            row_idx = int(img_id.replace("img_", ""))
            if row_idx < len(df):
                comment = str(df.iloc[row_idx]["comment"])[:300]  # store more, truncate later
                img_path = os.path.join(_project_root, "data", "images", f"{img_id}.jpg")
                if os.path.exists(img_path):
                    self.pairs.append({
                        "text": comment, "image_path": img_path, "img_id": img_id,
                        "room_type": meta.get("room_type", ""),
                        "comment_id": meta.get("comment_id", ""),
                    })
        if len(self.pairs) > max_pairs:
            self.pairs = self.pairs[:max_pairs]
        print(f"  Dataset ready: {len(self.pairs)} pairs (aug={cfg.use_augmentation})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = Image.open(p["image_path"]).convert("RGB")
        if self.cfg.use_augmentation and _has_torchvision:
            img = _aug_transform(img)
        text = p["text"]
        if self.cfg.use_augmentation:
            # 随机截断文本作为增强
            min_len = min(30, len(text))
            max_len = len(text)
            if max_len > min_len:
                cut = random.randint(min_len, max_len)
                text = text[:cut]
        return {"text": text, "image": img,
                "img_id": p["img_id"], "comment_id": p["comment_id"], "room_type": p["room_type"]}


def collate(batch, processor, max_length: int = 77):
    """Collate with configurable max_length."""
    texts = [b["text"] for b in batch]
    images = [b["image"] for b in batch]
    # 如果是 augmented tensor，需要特殊处理
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
        ii = processor(images=images, return_tensors="pt", do_rescale=True)
    else:
        ii = processor(images=images, return_tensors="pt")
    ti = processor(text=texts, return_tensors="pt", padding=True,
                   truncation=True, max_length=max_length)
    return ti, ii


# ═══════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════

def clip_loss(text_embs, image_embs, temperature=0.07, label_smoothing=0.0,
              i2t_weight=1.0, hard_neg_mask=None, hard_neg_boost=0.0):
    """对称 CLIP loss with optional enhancements."""
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
    logits = text_embs @ image_embs.T / temperature
    B = logits.shape[0]

    # Hard negative boost
    if hard_neg_mask is not None and hard_neg_boost > 0:
        logits = logits + hard_neg_mask * hard_neg_boost

    labels = torch.arange(B, device=logits.device)

    if label_smoothing > 0:
        smooth = label_smoothing / (B - 1)
        target = torch.full_like(logits, smooth)
        target.fill_diagonal_(1.0 - label_smoothing)
        loss_t2i = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        loss_i2t = -(target * F.log_softmax(logits.T, dim=-1)).sum(dim=-1).mean()
    else:
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.T, labels)

    w_sum = 1.0 + i2t_weight
    return (loss_t2i + i2t_weight * loss_i2t) / w_sum


# ═══════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════

def build_hard_neg_mask(embs, comment_ids, device):
    """Pre-compute hard negative mask for the current batch."""
    B = len(comment_ids)
    mask = torch.zeros(B, B, device=device)
    for i in range(B):
        for j in range(B):
            if i != j and comment_ids[i] != comment_ids[j]:
                sim = float(torch.dot(embs[i], embs[j]))
                if sim > 0.85:  # threshold for "hard"
                    mask[i, j] = 1.0
    return mask


def train(cfg: ExperimentConfig):
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    print(f"Experiment: {cfg.name}")
    print(f"Device: {DEVICE} | epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr}")
    print(f"Config: max_len={cfg.max_length} temp={cfg.temperature} "
          f"smooth={cfg.label_smoothing} i2t_w={cfg.i2t_weight} "
          f"aug={cfg.use_augmentation} hn_boost={cfg.hard_neg_boost}")

    print("Loading Chinese-CLIP...")
    model = ChineseCLIPModel.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True).to(DEVICE)
    processor = ChineseCLIPProcessor.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True)

    # Freeze based on trainable_patterns
    for name, param in model.named_parameters():
        should_train = any(p in name for p in cfg.trainable_patterns)
        param.requires_grad = should_train
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    store = NumpyVectorStore(os.path.join(_project_root, "data", "vectors_np"))
    df = pd.read_parquet(os.path.join(_project_root, "data", "hotel_reviews_table.parquet"))
    dataset = PairDataset(processor, df, store, cfg)

    # Learnable temperature
    if cfg.learnable_temp:
        logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / cfg.temperature), device=DEVICE))
        optimizer = torch.optim.AdamW(
            list([p for p in model.parameters() if p.requires_grad]) + [logit_scale],
            lr=cfg.lr)
    else:
        logit_scale = None
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    print(f"\nTraining {cfg.epochs} epochs...")
    model.train()
    history = []

    for epoch in range(cfg.epochs):
        t0 = time.time()
        indices = np.random.permutation(len(dataset))
        n_batches = len(indices) // cfg.batch_size
        total_loss = 0.0

        for b in range(n_batches):
            batch = [dataset[i] for i in indices[b*cfg.batch_size:(b+1)*cfg.batch_size]]
            ti, ii = collate(batch, processor, max_length=cfg.max_length)
            ti = {k: v.to(DEVICE) for k, v in ti.items()}
            ii = {k: v.to(DEVICE) for k, v in ii.items()}

            # Backbone mode
            if cfg.backbone_in_eval:
                model.text_model.eval()
                model.vision_model.eval()

            # Manual forward
            t_out = model.text_model(**ti)
            t_feats = model.text_projection(t_out.last_hidden_state[:, 0, :])
            i_out = model.vision_model(**ii)
            i_feats = model.visual_projection(i_out.last_hidden_state[:, 0, :])

            # Hard negative mask
            hn_mask = None
            if cfg.hard_neg_boost > 0:
                with torch.no_grad():
                    t_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)
                    i_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
                    cids = [batch[i]["comment_id"] for i in range(len(batch))]
                    hn_mask = build_hard_neg_mask(t_norm, cids, DEVICE)

            temp = cfg.temperature if logit_scale is None else logit_scale.exp().item()
            loss = clip_loss(t_feats, i_feats, temperature=temp,
                           label_smoothing=cfg.label_smoothing,
                           i2t_weight=cfg.i2t_weight,
                           hard_neg_mask=hn_mask,
                           hard_neg_boost=cfg.hard_neg_boost)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Eval on batch subset
        if epoch % 2 == 0 or epoch == cfg.epochs - 1:
            with torch.no_grad():
                model.eval()
                sample = [dataset[i] for i in range(min(50, len(dataset)))]
                ti, ii = collate(sample, processor, max_length=cfg.max_length)
                ti = {k: v.to(DEVICE) for k, v in ti.items()}
                ii = {k: v.to(DEVICE) for k, v in ii.items()}
                t_out = model.text_model(**ti)
                tf = model.text_projection(t_out.last_hidden_state[:, 0, :])
                i_out = model.vision_model(**ii)
                inf = model.visual_projection(i_out.last_hidden_state[:, 0, :])
                tf_n = tf / tf.norm(dim=-1, keepdim=True)
                inf_n = inf / inf.norm(dim=-1, keepdim=True)
                sim = tf_n @ inf_n.T
                t2i_hit = (sim.argmax(dim=-1) == torch.arange(
                    len(tf), device=sim.device)).float().mean().item()
                diag = (tf_n * inf_n).sum(dim=-1).mean().item()
            model.train()
        else:
            t2i_hit, diag = 0.0, 0.0

        history.append({"epoch": epoch+1, "loss": round(avg_loss, 4),
                       "t2i_hit": round(t2i_hit, 4), "diag_cos": round(diag, 4)})
        extra = f" | t2i_hit={t2i_hit:.2%} diag_cos={diag:.3f}" if t2i_hit > 0 else ""
        print(f"  Epoch {epoch+1}/{cfg.epochs} | loss={avg_loss:.4f}{extra} | {time.time()-t0:.0f}s")

    # Save to experiment directory
    save_dir = os.path.join(_project_root, "data", "experiments", cfg.name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "text_projection": model.text_projection.state_dict(),
        "visual_projection": model.visual_projection.state_dict(),
    }, os.path.join(save_dir, "projection_heads.pt"))

    report = {
        "config": {k: str(v) if isinstance(v, list) else v
                   for k, v in cfg.__dict__.items()},
        "trainable_params": trainable,
        "final_loss": history[-1]["loss"],
        "final_t2i_hit": history[-1]["t2i_hit"],
        "final_diag_cos": history[-1]["diag_cos"],
        "history": history,
    }
    with open(os.path.join(save_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {save_dir}")
    print(f"T2I Hit: {history[0].get('t2i_hit', 0):.2%} -> {history[-1]['t2i_hit']:.2%}")

    # Also save as default if baseline
    if cfg.name == "baseline":
        default_dir = os.path.join(_project_root, "data", "finetuned")
        os.makedirs(default_dir, exist_ok=True)
        torch.save({
            "text_projection": model.text_projection.state_dict(),
            "visual_projection": model.visual_projection.state_dict(),
        }, os.path.join(default_dir, "projection_heads.pt"))

    return model, processor


# ═══════════════════════════════════════════════════
# Rebuild Index
# ═══════════════════════════════════════════════════

def rebuild_index(exp_name: str = "baseline"):
    """用实验的投影头重建 NumPy 索引."""
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    # Load from experiment directory
    exp_dir = os.path.join(_project_root, "data", "experiments", exp_name)
    ft_path = os.path.join(exp_dir, "projection_heads.pt")
    if not os.path.exists(ft_path):
        ft_path = os.path.join(_project_root, "data", "finetuned", "projection_heads.pt")

    cfg = PRESETS.get(exp_name, ExperimentConfig(name=exp_name))
    max_length = cfg.max_length

    print(f"Rebuilding index with {exp_name} projections (max_length={max_length})...")
    model = ChineseCLIPModel.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True).eval().to(DEVICE)
    processor = ChineseCLIPProcessor.from_pretrained(
        "OFA-Sys/chinese-clip-vit-base-patch16", local_files_only=True)

    if os.path.exists(ft_path):
        state = torch.load(ft_path, map_location=DEVICE, weights_only=True)
        model.text_projection.load_state_dict(state["text_projection"])
        model.visual_projection.load_state_dict(state["visual_projection"])
        print(f"  [{exp_name} projections loaded]")

    df = pd.read_parquet(os.path.join(_project_root, "data", "hotel_reviews_table.parquet"))
    store = NumpyVectorStore(os.path.join(_project_root, "data", "vectors_np"))
    img_dir = os.path.join(_project_root, "data", "images")

    # Texts
    print("Rebuilding text index (2000)...")
    df_u = df.drop_duplicates(subset="comment").head(2000)
    t_ids, t_embs, t_metas, t_docs = [], [], [], []
    for start in range(0, len(df_u), 32):
        batch = df_u.iloc[start:start+32]
        texts = [str(r["comment"])[:300] for _, r in batch.iterrows()]
        ids = [str(r["_id"]) for _, r in batch.iterrows()]
        metas = [{"comment_id": str(r["_id"]), "content": t, "score": float(r["score"]),
                  "room_type": str(r.get("fuzzy_room_type", ""))}
                 for (_, r), t in zip(batch.iterrows(), texts)]
        ti = processor(text=texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length)
        with torch.no_grad():
            t_out = model.text_model(**{k: v.to(DEVICE) for k, v in ti.items()})
            feats = model.text_projection(t_out.last_hidden_state[:, 0, :])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        t_ids.extend(ids); t_embs.append(feats.cpu().numpy().astype(np.float32))
        t_metas.extend(metas); t_docs.extend(texts)
        if start % 640 == 0:
            print(f"  Texts: {min(start+32, len(df_u))}/{len(df_u)}")
    store.create("hotel_comments", t_ids, np.concatenate(t_embs, axis=0), t_metas, t_docs)

    # Images
    print("Rebuilding image index...")
    img_data = store.get("hotel_images", include=["metadatas"])
    i_ids, i_embs, i_metas, i_docs = [], [], [], []
    for i, (img_id, meta) in enumerate(zip(img_data["ids"], img_data["metadatas"])):
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        ii = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            i_out = model.vision_model(**{k: v.to(DEVICE) for k, v in ii.items()})
            feat = model.visual_projection(i_out.last_hidden_state[:, 0, :])
        feat = feat / feat.norm(dim=-1, keepdim=True)
        i_ids.append(img_id); i_embs.append(feat.squeeze().cpu().numpy().astype(np.float32))
        i_metas.append(meta); i_docs.append(meta.get("comment", ""))
        if (i+1) % 200 == 0:
            print(f"  Images: {i+1}/{len(img_data['ids'])}")
    store.create("hotel_images", i_ids, np.array(i_embs, dtype=np.float32), i_metas, i_docs)
    print(f"Done: {store.count('hotel_images')} images, {store.count('hotel_comments')} texts")


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline",
                       choices=list(PRESETS.keys()),
                       help="实验名称 (预设配置)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="覆盖预设的 epochs")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--list", action="store_true",
                       help="列出所有可用的实验")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, cfg in PRESETS.items():
            print(f"  {name}:")
            print(f"    trainable: {cfg.trainable_patterns}")
            print(f"    aug={cfg.use_augmentation} temp={cfg.temperature} "
                  f"smooth={cfg.label_smoothing} i2t_w={cfg.i2t_weight} "
                  f"max_len={cfg.max_length} hn={cfg.hard_neg_boost}")
        sys.exit(0)

    if args.rebuild_only:
        rebuild_index(args.experiment)
    else:
        cfg = PRESETS[args.experiment]
        # Override with CLI args
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        if args.lr is not None:
            cfg.lr = args.lr

        print("=" * 50)
        print(f"Experiment: {cfg.name}")
        print("=" * 50)
        train(cfg)
        print("\n" + "=" * 50)
        print("Rebuilding index...")
        print("=" * 50)
        rebuild_index(cfg.name)
        print("\nDone.")
