"""
评测迭代执行器 v2 — 遵循 5 步方法论 + 跨模态/难负样本扩展.

用法:
    python experiments.py --task ltr --round 4    # 跨模态特征
    python experiments.py --task ltr --round 5    # 难负样本加权
    python experiments.py --task ltr --report
"""
import sys, os, json, time, warnings, random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd

warnings.filterwarnings("ignore")

# ⚠️ ARCHIVED — 评测迭代实验，不进入生产调用链
_this_dir = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_dir))))
sys.path.insert(0, os.path.join(_PROJ, "scripts"))

from leuca.multimodal.api import MultimodalAPI

DEVICE = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

api = MultimodalAPI()
store = api.retriever.store
df = api.retriever.comments
LOG_PATH = os.path.join(_this_dir, "iteration_log.json")


# ═══════════════════════════════════════════════
# Configs
# ═══════════════════════════════════════════════

@dataclass
class LTRConfig:
    name: str; description: str
    n_train_queries: int = 200; n_test_queries: int = 100
    features: List[str] = field(default_factory=lambda: [
        "clip_cos", "same_room", "score_drop", "norm_rank"])
    hidden_dim: int = 32; epochs: int = 30; lr: float = 1e-3
    loss_type: str = "pairwise"
    cross_modal: bool = False
    hard_neg_weight: float = 0.0

LTR_PRESETS = {
    0: LTRConfig(name="R0_CLIP_baseline", description="CLIP only, no re-ranking"),
    1: LTRConfig(name="R1_MLP_4feat_pairwise",
                 description="MLP 4feat pairwise loss, 200q, 30ep"),
    2: LTRConfig(name="R2_5feat_listwise_500q",
                 description="5feat listwise loss, 500q, 20ep",
                 n_train_queries=500,
                 features=["clip_cos","same_room","score_drop","norm_rank","score_std"],
                 epochs=20, loss_type="listwise"),
    3: LTRConfig(name="R3_7feat_deeper_500q",
                 description="7feat deeper MLP (64), listwise, 500q, 25ep",
                 n_train_queries=500, hidden_dim=64,
                 features=["clip_cos","same_room","score_drop","norm_rank",
                          "score_std","comment_len","query_room_has"],
                 epochs=25, loss_type="listwise"),
    4: LTRConfig(name="R4_cross_modal_fusion",
                 description="Cross-modal features + listwise, 500q, 20ep (v5 #1)",
                 n_train_queries=500,
                 features=["clip_cos","same_room","score_drop","norm_rank",
                          "score_std","cross_text_sim","cross_rank_ratio"],
                 epochs=20, loss_type="listwise", cross_modal=True),
    5: LTRConfig(name="R5_hard_neg_weighted",
                 description="Hard negative weighting + listwise, 500q, 20ep (v5 #4)",
                 n_train_queries=500,
                 features=["clip_cos","same_room","score_drop","norm_rank",
                          "score_std","cross_text_sim","cross_rank_ratio"],
                 epochs=20, loss_type="listwise", cross_modal=True,
                 hard_neg_weight=3.0),
}


# ═══════════════════════════════════════════════
# 数据构建
# ═══════════════════════════════════════════════

def load_split():
    split_dir = os.path.join(_PROJ, "data", "split")
    with open(os.path.join(split_dir, "train_ids.json")) as f:
        train_ids = set(json.load(f))
    with open(os.path.join(split_dir, "test_ids.json")) as f:
        test_ids = set(json.load(f))
    return train_ids, test_ids


def build_ltr_samples(subset_ids: set, n_queries: int, features: List[str],
                      cross_modal: bool = False):
    """构建 LTR 样本，可选跨模态特征."""
    all_img_ids = store.get("hotel_images", include=["metadatas"])
    img_metas = {iid: m for iid, m in zip(all_img_ids["ids"], all_img_ids["metadatas"])}
    subset_list = sorted([iid for iid in subset_ids if iid in img_metas])[:n_queries]

    samples = []
    for i, img_id in enumerate(subset_list):
        meta = img_metas[img_id]
        row_idx = int(img_id.replace("img_", ""))
        if row_idx >= len(df): continue
        comment = str(df.iloc[row_idx]["comment"])[:200]
        gt_img = f"data/images/{img_id}.jpg"
        query_room = meta.get("room_type", "")

        result = api.prompt2image(comment, topK=20)
        retrieved = result["images"]; scores = result["score"]

        # Cross-modal: 对每个候选图片做 I→T 检索，获得"反向描述"
        cross_texts = {}
        if cross_modal:
            for ret_img in retrieved[:5]:  # top-5 候选做反向检索
                try:
                    i2t = api.image2text(ret_img, topK=3)
                    cross_texts[ret_img] = i2t["texts"][0][:100] if i2t["texts"] else ""
                except Exception:
                    cross_texts[ret_img] = ""

        for rank, (ret_img, score) in enumerate(zip(retrieved, scores)):
            ret_id = ret_img.split("/")[-1].replace(".jpg", "")
            ret_meta = img_metas.get(ret_id, {})
            ret_room = ret_meta.get("room_type", "")
            is_hit = 1 if ret_img == gt_img else 0

            # Cross-modal features
            cross_text_sim = 0.0
            cross_rank_ratio = 0.0
            if cross_modal and ret_img in cross_texts:
                ct = cross_texts[ret_img]
                if ct and comment:
                    # 简单文本重叠度
                    words_q = set(comment[:80])
                    words_t = set(ct[:80])
                    cross_text_sim = len(words_q & words_t) / max(len(words_q | words_t), 1)

            feat_dict = {
                "clip_cos": score,
                "same_room": 1.0 if ret_room == query_room else 0.0,
                "score_drop": scores[0] - score if rank > 0 else 0.0,
                "norm_rank": rank / 20.0,
                "score_std": float(np.std(scores[:5])),
                "comment_len": min(len(comment) / 300.0, 1.0),
                "query_room_has": 1.0 if query_room in comment else 0.0,
                "cross_text_sim": cross_text_sim,
                "cross_rank_ratio": cross_rank_ratio,
            }
            feat_vec = [feat_dict[f] for f in features]

            # Hard negative label (同房型但不同评论 = hard negative)
            is_hard_neg = (not is_hit) and (ret_room == query_room)

            samples.append({
                "query_id": img_id, "features": feat_vec, "label": is_hit,
                "retrieved": ret_img, "is_hard_neg": is_hard_neg,
            })

        if (i+1) % 100 == 0:
            print(f"    Queries: {i+1}/{len(subset_list)}")

    hits = sum(s["label"] for s in samples)
    hns = sum(s["is_hard_neg"] for s in samples)
    print(f"    Samples: {len(samples)}, hits: {hits} ({hits/len(samples):.3%}), "
          f"hard_negs: {hns}")
    return samples


# ═══════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════

class LTRModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        layers = []
        dims = [n_features, hidden, hidden, 1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_ltr_round(samples, config: LTRConfig):
    n_feat = len(config.features)
    X = torch.tensor([s["features"] for s in samples], dtype=torch.float32).to(DEVICE)
    y = torch.tensor([s["label"] for s in samples], dtype=torch.float32).to(DEVICE)
    is_hn = torch.tensor([float(s["is_hard_neg"]) for s in samples],
                         dtype=torch.float32).to(DEVICE)

    query_groups = defaultdict(list)
    for i, s in enumerate(samples):
        query_groups[s["query_id"]].append(i)

    model = LTRModel(n_feat, config.hidden_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    print(f"  Training ({len(samples)} samples, {n_feat} feat, "
          f"{config.loss_type}, cross_modal={config.cross_modal}, "
          f"hn_weight={config.hard_neg_weight})...")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0

        if config.loss_type == "pairwise":
            n_pairs = 0
            for indices in query_groups.values():
                if len(indices) < 2: continue
                scores = model(X[indices])
                for a in range(len(indices)):
                    for b in range(len(indices)):
                        if y[indices[a]] > y[indices[b]]:
                            w = 1.0
                            if config.hard_neg_weight > 0 and is_hn[indices[b]]:
                                w = config.hard_neg_weight
                            total_loss += w * F.relu(1.0 - (scores[a] - scores[b]))
                            n_pairs += w
            total_loss = total_loss / max(n_pairs, 1)

        elif config.loss_type == "listwise":
            for indices in query_groups.values():
                if len(indices) < 2: continue
                logits = model(X[indices]); labels = y[indices]
                probs = F.softmax(logits / 0.1, dim=0)
                target = labels / (labels.sum() + 1e-8)
                # Apply hard negative weight
                if config.hard_neg_weight > 0:
                    weights = 1.0 + is_hn[indices] * (config.hard_neg_weight - 1.0)
                    loss = -(weights * target * torch.log(probs + 1e-8)).sum()
                else:
                    loss = -(target * torch.log(probs + 1e-8)).sum()
                total_loss += loss
            total_loss = total_loss / max(len(query_groups), 1)

        opt.zero_grad(); total_loss.backward(); opt.step()

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            with torch.no_grad():
                model.eval(); preds = model(X)
                correct = total = 0
                for indices in query_groups.values():
                    if len(indices) < 2: continue
                    ps = preds[indices]; ys = y[indices]
                    for a in range(len(indices)):
                        for b in range(len(indices)):
                            if ys[a] > ys[b]:
                                correct += (ps[a] > ps[b]).float().item(); total += 1
            print(f"    E{epoch+1}/{config.epochs} loss={total_loss.item():.4f} "
                  f"pair_acc={correct/max(total,1):.3f}")

    return model


def evaluate_ltr_round(model, samples, name: str):
    model.eval()
    query_groups = defaultdict(list)
    for i, s in enumerate(samples):
        query_groups[s["query_id"]].append(i)

    clip_hits = {1:0,5:0,10:0}; ltr_hits = {1:0,5:0,10:0}; n=0
    with torch.no_grad():
        for indices in query_groups.values():
            items = [samples[i] for i in indices]
            clip_ranked = sorted(items, key=lambda s: -s["features"][0])
            X = torch.tensor([s["features"] for s in items], dtype=torch.float32).to(DEVICE)
            ltr_scores = model(X).cpu().numpy()
            ltr_ranked = [items[i] for i in np.argsort(-ltr_scores)]
            n += 1
            for k in [1,5,10]:
                clip_hits[k] += any(s["label"] for s in clip_ranked[:k])
                ltr_hits[k] += any(s["label"] for s in ltr_ranked[:k])

    return {
        "clip": {f"hit{k}": clip_hits[k]/n for k in [1,5,10]},
        "ltr": {f"hit{k}": ltr_hits[k]/n for k in [1,5,10]},
    }


# ═══════════════════════════════════════════════
# 执行
# ═══════════════════════════════════════════════

def run_ltr_round(round_num: int):
    config = LTR_PRESETS[round_num]
    print(f"\n{'='*60}")
    print(f"LTR Round {round_num}: {config.name}")
    print(f"  {config.description}")
    print(f"{'='*60}")

    train_ids, test_ids = load_split()

    if round_num == 0:
        test_samples = build_ltr_samples(test_ids, config.n_test_queries, config.features)
        query_groups = defaultdict(list)
        for i, s in enumerate(test_samples):
            query_groups[s["query_id"]].append(i)
        n = 0; clip_hits = {1:0,5:0,10:0}
        for indices in query_groups.values():
            items = [test_samples[i] for i in indices]
            ranked = sorted(items, key=lambda s: -s["features"][0]); n += 1
            for k in [1,5,10]:
                clip_hits[k] += any(s["label"] for s in ranked[:k])
        metrics = {"clip": {f"hit{k}": clip_hits[k]/n for k in [1,5,10]}}
    else:
        print(f"  Building {config.n_train_queries} train + {config.n_test_queries} test samples...")
        if config.cross_modal:
            print(f"  (Cross-modal feature computation enabled — slower)")
        train_samples = build_ltr_samples(train_ids, config.n_train_queries,
                                          config.features, config.cross_modal)
        test_samples = build_ltr_samples(test_ids, config.n_test_queries,
                                         config.features, config.cross_modal)
        model = train_ltr_round(train_samples, config)
        metrics = evaluate_ltr_round(model, test_samples, config.name)

    return {
        "round": round_num, "name": config.name, "description": config.description,
        "config": {k: str(v) if isinstance(v, list) else v
                   for k, v in config.__dict__.items()},
        "metrics": metrics,
    }


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tasks": {}}


def save_log(log):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def print_ltr_history(log):
    rounds = log["tasks"].get("ltr", {}).get("rounds", [])
    if not rounds: return
    print(f"\n{'='*70}")
    print(f"LTR Iteration History")
    print(f"{'='*70}")
    print(f"  {'R':<4} {'Name':<28} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'Δ vs R0':>8}")
    print(f"  {'-'*4} {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    r0_h5 = rounds[0]["metrics"].get("ltr", rounds[0]["metrics"].get("clip",{})).get("hit5", 0)
    for r in rounds:
        m = r.get("metrics", {}).get("ltr", r.get("metrics", {}).get("clip", {}))
        h5 = m.get("hit5", 0)
        delta = f"+{h5-r0_h5:.4f}" if h5 >= r0_h5 else f"{h5-r0_h5:.4f}"
        mark = " < best" if h5 == max(rr.get("metrics",{}).get("ltr",rr.get("metrics",{}).get("clip",{})).get("hit5",0) for rr in rounds) else ""
        print(f"  R{r['round']:<3} {r['name']:<28} {m.get('hit1',0):>7.4f} {h5:>7.4f} {m.get('hit10',0):>7.4f} {delta:>8}{mark}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="ltr", choices=["ltr"])
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    log = load_log()
    if "ltr" not in log["tasks"]:
        log["tasks"]["ltr"] = {"rounds": []}

    if args.report:
        print_ltr_history(log)
        sys.exit(0)

    rounds_to_run = [args.round] if args.round is not None else sorted(LTR_PRESETS.keys())
    for r in rounds_to_run:
        result = run_ltr_round(r)
        existing = [e for e in log["tasks"]["ltr"]["rounds"] if e["round"] != r]
        existing.append(result); existing.sort(key=lambda x: x["round"])
        log["tasks"]["ltr"]["rounds"] = existing
        save_log(log)

    print_ltr_history(log)
    print(f"\nLog: {LOG_PATH}")
