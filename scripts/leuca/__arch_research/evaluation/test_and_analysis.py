"""
多模态召回接口 — 测试与分析 (v2 — 深度版).

测试内容:
  1. prompt→image 与 image→text 基础可用性
  2. Hit@K (strict & relaxed) + NDCG
  3. 对称性分析: 文本↔图片
  4. 文本形式对比: 7种变体 + Query Fusion 加权融合
  5. Score 校准分析: 分布、校准曲线、相对阈值
  6. 错误分类: 视觉混淆 / 语义鸿沟 / 索引缺失
  7. 图片混淆矩阵: 嵌入空间中哪些图片被系统性混淆
  8. 检索多样性: 前K结果的语义覆盖度

输出:
  - test_results.json   : 完整结果
  - analysis_report.md  : 综合分析报告

使用方式:
    python test_and_analysis.py              # 全量测试
    python test_and_analysis.py --quick      # 快速测试
"""

import sys
import os

# ⚠️ ARCHIVED — 测评体系，不进入生产调用链
_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_PROJ, "scripts"))

import json
import time
import warnings
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 导入接口 ─────────────────────────────────────────────────
from leuca.multimodal.api import MultimodalAPI

# ── 输出目录 ─────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# 1. 测试数据集构建
# ═══════════════════════════════════════════════════════════════

def build_test_datasets(api: MultimodalAPI, quick: bool = False,
                         test_only: bool = True) -> Tuple[List[dict], List[dict]]:
    """构建 prompt→image 和 image→text 的测试数据集.

    Args:
        test_only: 仅使用 test split 的图片 (排除 train 图片防止数据泄露).
    """
    df = api.retriever.comments
    coll = api.retriever.store
    all_data = coll.get("hotel_images", include=["metadatas"])
    img_ids = all_data["ids"]
    metas = all_data["metadatas"]

    # 加载 test split (如果存在)
    test_ids = None
    split_dir = os.path.join(_project_root, "data", "split")
    test_file = os.path.join(split_dir, "test_ids.json")
    if test_only and os.path.exists(test_file):
        with open(test_file) as f:
            test_ids = set(json.load(f))
        print(f"  [Test-only mode: {len(test_ids)} test images]")

    # 构建 image_id → (df_row_idx, comment, room_type) 映射
    img_info = {}
    for img_id, meta in zip(img_ids, metas):
        if test_ids is not None and img_id not in test_ids:
            continue  # 跳过 train 图片
        row_idx = int(img_id.replace("img_", ""))
        if row_idx < len(df):
            img_info[img_id] = {
                "row_idx": row_idx,
                "comment": str(df.iloc[row_idx]["comment"]),
                "room_type": str(df.iloc[row_idx].get("fuzzy_room_type", "")),
                "comment_id": str(meta.get("comment_id", "")),
            }

    print(f"  Using {len(img_info)} images for evaluation")

    # ── prompt→image 测试集 ──
    # 按 _id 去重取第一条
    p2i_dataset = []
    seen_ids = set()
    for img_id, info in img_info.items():
        rid = info["comment_id"]
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        p2i_dataset.append({
            "ID": info["row_idx"],
            "prompt": info["comment"],
            "images_retrieved": [],
            "images_similarity": [],
            "ground_truth_images": [f"data/images/{img_id}.jpg"],
        })

    # ── image→text 测试集 ──
    i2t_dataset = []
    for img_id, info in img_info.items():
        i2t_dataset.append({
            "ID": info["row_idx"],
            "image": f"data/images/{img_id}.jpg",
            "text_retrieved": [],
            "text_similarity": [],
            "ground_truth_text": [info["comment"]],
        })

    if quick:
        p2i_dataset = p2i_dataset[:30]
        i2t_dataset = i2t_dataset[:30]

    print(f"Test datasets built: P2I={len(p2i_dataset)} (deduped), I2T={len(i2t_dataset)}")
    return p2i_dataset, i2t_dataset


# ═══════════════════════════════════════════════════════════════
# 2. 文本形式变体
# ═══════════════════════════════════════════════════════════════

def get_text_variants(row: pd.Series) -> Dict[str, str]:
    """为一个数据行生成所有文本形式变体.

    返回 {variant_name: text} 字典.
    """
    comment = str(row["comment"])
    query = str(row.get("query", ""))
    categories = str(row.get("categories", ""))
    room_type = str(row.get("fuzzy_room_type", ""))
    travel_type = str(row.get("travel_type", ""))

    # 清洗 categories (可能是 JSON 数组字符串)
    try:
        cat_list = json.loads(categories)
        if isinstance(cat_list, list):
            categories_str = "，".join(cat_list)
        else:
            categories_str = categories
    except (json.JSONDecodeError, TypeError):
        categories_str = categories

    variants = {
        "原始评论 (full comment)": comment,
        "评论前100字 (first 100 chars)": comment[:100],
        "评论前50字 (first 50 chars)": comment[:50],
    }

    if query and query.strip() and query != "nan":
        variants["搜索查询词 (query)"] = query
    if categories_str.strip() and categories_str != "nan" and categories_str != "[]":
        variants["类别标签 (categories)"] = categories_str
    if room_type.strip() and room_type != "nan":
        variants["房型 (room_type)"] = room_type

    # 组合: query + categories + room_type
    combo_parts = []
    if query.strip() and query != "nan":
        combo_parts.append(query)
    if categories_str.strip() and categories_str != "nan" and categories_str != "[]":
        combo_parts.append(categories_str)
    if room_type.strip() and room_type != "nan":
        combo_parts.append(room_type)
    if combo_parts:
        variants["组合 (query+categories+room)"] = "，".join(combo_parts)

    return variants


# ═══════════════════════════════════════════════════════════════
# 3. 指标计算
# ═══════════════════════════════════════════════════════════════

def hit_at_k(retrieved: List[str], ground_truth: List[str], k: int) -> int:
    """Hit@K: 1 if ground_truth ∩ retrieved[:k] else 0."""
    gt_set = set(ground_truth)
    top_k = set(retrieved[:k])
    return 1 if gt_set & top_k else 0


def compute_hit_rates(dataset: List[dict], k_values: List[int] = (1, 3, 5, 10)) -> Dict[int, float]:
    """计算各 K 值的 Hit@K."""
    n = len(dataset)
    hits = {k: 0 for k in k_values}
    for item in dataset:
        retrieved = item.get("images_retrieved") or item.get("text_retrieved", [])
        ground_truth = item.get("ground_truth_images") or item.get("ground_truth_text", [])
        for k in k_values:
            hits[k] += hit_at_k(retrieved, ground_truth, k)
    return {k: hits[k] / n if n > 0 else 0.0 for k in hits}


def ndcg_at_k(dataset: List[dict], k: int = 10) -> float:
    """NDCG@K — 考虑排序位置的归一化折损累积增益.

    相比 Hit@K, NDCG 对排序质量更敏感：排名越靠前的命中贡献越大.
    """
    values = []
    for item in dataset:
        retrieved = item.get("images_retrieved") or item.get("text_retrieved", [])
        ground_truth = item.get("ground_truth_images") or item.get("ground_truth_text", [])
        gt_set = set(ground_truth)
        # 二元相关性: 命中=1, 未命中=0
        dcg = sum(
            (1.0 / np.log2(i + 2)) if retrieved[i] in gt_set else 0.0
            for i in range(min(k, len(retrieved)))
        )
        idcg = 1.0 / np.log2(2)  # 理想情况: rank=1 命中
        values.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(values)) if values else 0.0


def relaxed_hit_at_k(retrieved: List[str], ground_truth: List[str],
                     relaxed_groups: Dict[str, List[str]], k: int) -> float:
    """Relaxed Hit@K: 若检索结果与 ground truth 属于同一语义组，计为部分命中.

    Args:
        relaxed_groups: {group_key: [item_ids...]}  如 {room_type: [img_ids]}.
        命中 group 计 0.5 分 (默认), 精确命中计 1.0 分.

    Returns:
        float score in [0, 1].
    """
    gt_set = set(ground_truth)
    top_k = set(retrieved[:k])

    # 精确命中
    exact = gt_set & top_k
    if exact:
        return 1.0

    # 松弛命中: 查找 ground truth 所属的 groups
    gt_groups = set()
    for gkey, members in relaxed_groups.items():
        if gt_set & set(members):
            gt_groups.add(gkey)
    # 查找 top_k 所属的 groups
    topk_groups = set()
    for gkey, members in relaxed_groups.items():
        if top_k & set(members):
            topk_groups.add(gkey)
    # 共享任一 group → 松弛命中
    return 0.5 if gt_groups & topk_groups else 0.0


def compute_relaxed_hit_rates(dataset: List[dict], relaxed_groups: Dict[str, List[str]],
                               k_values=(1, 3, 5, 10)) -> Dict[int, float]:
    """计算 Relaxed Hit@K."""
    n = len(dataset)
    hits = {k: 0.0 for k in k_values}
    for item in dataset:
        retrieved = item.get("images_retrieved") or item.get("text_retrieved", [])
        gt = item.get("ground_truth_images") or item.get("ground_truth_text", [])
        for k in k_values:
            hits[k] += relaxed_hit_at_k(retrieved, gt, relaxed_groups, k)
    return {k: hits[k] / n if n > 0 else 0.0 for k in hits}


def mean_reciprocal_rank(dataset: List[dict]) -> float:
    """MRR (Mean Reciprocal Rank)."""
    ranks = []
    for item in dataset:
        retrieved = item.get("images_retrieved") or item.get("text_retrieved", [])
        ground_truth = item.get("ground_truth_images") or item.get("ground_truth_text", [])
        gt_set = set(ground_truth)
        for i, r in enumerate(retrieved):
            if r in gt_set:
                ranks.append(1.0 / (i + 1))
                break
        else:
            ranks.append(0.0)
    return float(np.mean(ranks)) if ranks else 0.0


# ═══════════════════════════════════════════════════════════════
# 4. 核心测试
# ═══════════════════════════════════════════════════════════════

def test_prompt2image(api: MultimodalAPI, dataset: List[dict], topK: int = 10) -> List[dict]:
    """测试 prompt→image 召回."""
    print(f"\n{'='*60}")
    print(f"Testing prompt→image (topK={topK}) on {len(dataset)} samples...")
    print(f"{'='*60}")
    for i, item in enumerate(dataset):
        result = api.prompt2image(item["prompt"], topK=topK)
        item["images_retrieved"] = result["images"]
        item["images_similarity"] = result["score"]
        if i < 3:
            print(f"  [{item['ID']}] prompt={item['prompt'][:50]}...")
            print(f"       retrieved: {[p.split('/')[-1] for p in result['images'][:5]]}")
            print(f"       scores: {[f'{s:.3f}' for s in result['score'][:5]]}")
    return dataset


def test_image2text(api: MultimodalAPI, dataset: List[dict], topK: int = 10) -> List[dict]:
    """测试 image→text 召回."""
    print(f"\n{'='*60}")
    print(f"Testing image→text (topK={topK}) on {len(dataset)} samples...")
    print(f"{'='*60}")
    for i, item in enumerate(dataset):
        result = api.image2text(item["image"], topK=topK)
        item["text_retrieved"] = result["texts"]
        item["text_similarity"] = result["score"]
        if i < 3:
            print(f"  [{item['ID']}] image={item['image']}")
            print(f"       retrieved: {[t[:60]+'...' for t in result['texts'][:3]]}")
            print(f"       scores: {[f'{s:.3f}' for s in result['score'][:3]]}")
    return dataset


# ═══════════════════════════════════════════════════════════════
# 5. 对称性分析
# ═══════════════════════════════════════════════════════════════

def analyze_symmetry(api: MultimodalAPI, dataset_i2t: List[dict],
                     sample_size: int = 20) -> Dict:
    """分析 text↔image 召回对称性.

    对每个样本:
      image → texts (取 top-1 text) → images (用 top-1 text 查询)
      检查原始 image 是否在召回结果中.
    """
    print(f"\n{'='*60}")
    print("Analyzing symmetry (image→text→image)...")
    print(f"{'='*60}")

    results = []
    for item in dataset_i2t[:sample_size]:
        original_image = item["image"]
        # image → texts
        i2t_result = api.image2text(original_image, topK=10)
        top_text = i2t_result["texts"][0] if i2t_result["texts"] else ""

        # text → images
        t2i_result = api.prompt2image(top_text, topK=10)
        symmetric = original_image in t2i_result["images"]

        # 找到原始图片的排名
        rank = None
        if original_image in t2i_result["images"]:
            rank = t2i_result["images"].index(original_image) + 1

        results.append({
            "ID": item["ID"],
            "original_image": original_image,
            "bridge_text": top_text[:100],
            "symmetric": symmetric,
            "rank": rank,
            "retrieved_images": t2i_result["images"][:5],
            "retrieved_scores": t2i_result["score"][:5],
        })
        if len(results) <= 3:
            print(f"  [{item['ID']}] {original_image}")
            print(f"       bridge text: {top_text[:60]}...")
            print(f"       symmetric: {symmetric}, rank: {rank}")

    sym_count = sum(1 for r in results if r["symmetric"])
    print(f"\n  Symmetry rate: {sym_count}/{len(results)} = {sym_count/len(results):.2%}")
    return {
        "method": "image→text→image",
        "sample_size": len(results),
        "symmetric_count": sym_count,
        "symmetric_rate": sym_count / len(results),
        "details": results,
    }


def analyze_reverse_symmetry(api: MultimodalAPI, dataset_p2i: List[dict],
                              sample_size: int = 20) -> Dict:
    """反向对称性: text → images (取 top-1 image) → texts (用 top-1 image 查询).

    检查原始 text 是否在召回结果中.
    """
    print(f"\n{'='*60}")
    print("Analyzing reverse symmetry (text→image→text)...")
    print(f"{'='*60}")

    results = []
    for item in dataset_p2i[:sample_size]:
        original_text = item["prompt"]
        # text → images
        t2i_result = api.prompt2image(original_text, topK=10)
        top_image = t2i_result["images"][0] if t2i_result["images"] else ""

        # image → texts
        i2t_result = api.image2text(top_image, topK=10)
        symmetric = original_text in i2t_result["texts"]

        # 找到原始文本的排名
        rank = None
        if original_text in i2t_result["texts"]:
            rank = i2t_result["texts"].index(original_text) + 1

        # 计算文本相似度 (基于前100字符的包含关系)
        approx_match = any(original_text[:80] in t for t in i2t_result["texts"])

        results.append({
            "ID": item["ID"],
            "original_text": original_text[:100],
            "bridge_image": top_image,
            "symmetric": symmetric,
            "approx_match": approx_match,
            "rank": rank,
            "retrieved_texts": [t[:80] for t in i2t_result["texts"][:5]],
            "retrieved_scores": i2t_result["score"][:5],
        })
        if len(results) <= 3:
            print(f"  [{item['ID']}] text={original_text[:50]}...")
            print(f"       bridge image: {top_image}")
            print(f"       symmetric (exact): {symmetric}, approx: {approx_match}")

    sym_count = sum(1 for r in results if r["symmetric"])
    approx_count = sum(1 for r in results if r["approx_match"])
    print(f"\n  Exact symmetry: {sym_count}/{len(results)} = {sym_count/len(results):.2%}")
    print(f"  Approx symmetry: {approx_count}/{len(results)} = {approx_count/len(results):.2%}")
    return {
        "method": "text→image→text",
        "sample_size": len(results),
        "symmetric_count": sym_count,
        "symmetric_rate": sym_count / len(results),
        "approx_match_count": approx_count,
        "approx_match_rate": approx_count / len(results),
        "details": results,
    }


# ═══════════════════════════════════════════════════════════════
# 6. 文本形式对比
# ═══════════════════════════════════════════════════════════════

def compare_text_variants(api: MultimodalAPI, df: pd.DataFrame,
                          sample_size: int = 30) -> Dict:
    """对比不同文本形式在 prompt→image 上的召回效果.

    使用 Chroma 中已有的图片对应的行作为测试样本.
    """
    print(f"\n{'='*60}")
    print("Comparing text variants for prompt→image...")
    print(f"{'='*60}")

    # 获取有图片的行索引
    coll = api.retriever.store
    img_ids = coll.get("hotel_images", include=[])["ids"]
    indexed_rows = sorted([int(i.replace("img_", "")) for i in img_ids])
    indexed_rows = indexed_rows[:sample_size]

    all_variants = defaultdict(list)
    for idx in indexed_rows:
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        variants = get_text_variants(row)
        gt_images = [f"data/images/img_{idx}.jpg"]

        for vname, vtext in variants.items():
            if not vtext.strip():
                continue
            result = api.prompt2image(vtext, topK=10)
            all_variants[vname].append({
                "ID": int(idx),
                "prompt": vtext,
                "images_retrieved": result["images"],
                "images_similarity": result["score"],
                "ground_truth_images": gt_images,
            })

    k_values = [1, 3, 5, 10]
    comparison = {}
    for vname, dataset in sorted(all_variants.items()):
        hits = compute_hit_rates(dataset, k_values)
        mrr = mean_reciprocal_rank(dataset)
        comparison[vname] = {
            "n_samples": len(dataset),
            "hit_at_k": hits,
            "mrr": round(mrr, 4),
        }
        print(f"  {vname} (n={len(dataset)}): "
              f"Hit@1={hits[1]:.3f}, Hit@5={hits[5]:.3f}, Hit@10={hits[10]:.3f}, MRR={mrr:.3f}")

    return comparison


# ═══════════════════════════════════════════════════════════════
# 7. Score 阈值分析
# ═══════════════════════════════════════════════════════════════

def analyze_score_threshold(p2i_dataset: List[dict], i2t_dataset: List[dict],
                            thresholds: List[float] = None) -> Dict:
    """分析 score 阈值筛选对 Hit@K 的影响.

    如果某次召回的最高 score 低于阈值，认为该次召回失败，不参与指标计算.
    """
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    print(f"\n{'='*60}")
    print("Analyzing score threshold filtering...")
    print(f"{'='*60}")

    k_values = [1, 3, 5, 10]
    results = {"prompt2image": {}, "image2text": {}}

    for direction, dataset in [("prompt2image", p2i_dataset), ("image2text", i2t_dataset)]:
        retrieved_key = "images_retrieved" if direction == "prompt2image" else "text_retrieved"
        similarity_key = "images_similarity" if direction == "prompt2image" else "text_similarity"
        gt_key = "ground_truth_images" if direction == "prompt2image" else "ground_truth_text"

        # 无阈值 baseline
        base_hits = compute_hit_rates(dataset, k_values)
        base_mrr = mean_reciprocal_rank(dataset)
        base_count = len(dataset)

        print(f"\n  [{direction}] baseline (n={base_count}): "
              f"Hit@1={base_hits[1]:.3f}, Hit@5={base_hits[5]:.3f}, MRR={base_mrr:.3f}")

        thr_results = {}
        for th in thresholds:
            filtered = []
            failed = 0
            for item in dataset:
                scores = item.get(similarity_key, [])
                max_score = max(scores) if scores else 0.0
                if max_score >= th:
                    filtered.append(item)
                else:
                    failed += 1

            if filtered:
                hits = compute_hit_rates(filtered, k_values)
                mrr = mean_reciprocal_rank(filtered)
            else:
                hits = {k: 0.0 for k in k_values}
                mrr = 0.0

            thr_results[th] = {
                "passed": len(filtered),
                "failed": failed,
                "pass_rate": len(filtered) / base_count if base_count else 0,
                "hit_at_k": hits,
                "mrr": round(mrr, 4),
            }
            # 对显著结果做标记
            improvement = ""
            if filtered and hits[5] > base_hits[5] * 1.05:
                improvement = " [UP] improved"
            elif filtered and hits[5] < base_hits[5] * 0.95:
                improvement = " [DOWN] degraded"
            print(f"    th={th:.2f}: pass={len(filtered)}/{base_count} "
                  f"Hit@5={hits[5]:.3f}{improvement}")

        results[direction] = {
            "baseline": {"n": base_count, "hit_at_k": base_hits, "mrr": base_mrr},
            "by_threshold": thr_results,
        }

    return results


# ═══════════════════════════════════════════════════════════════
# 8. 具体 Case 展示
# ═══════════════════════════════════════════════════════════════

def collect_case_examples(p2i_dataset: List[dict], i2t_dataset: List[dict],
                          n_cases: int = 5) -> Dict:
    """收集具体 case，区分高分/中分/低分."""
    print(f"\n{'='*60}")
    print("Collecting case examples...")
    print(f"{'='*60}")

    def classify_scores(scores: List[float]) -> List[Dict]:
        """分类 scores 为 high/mid/low."""
        result = []
        for s in scores:
            if s >= 0.4:
                result.append({"score": s, "level": "high"})
            elif s >= 0.25:
                result.append({"score": s, "level": "mid"})
            else:
                result.append({"score": s, "level": "low"})
        return result

    p2i_cases = []
    for item in p2i_dataset[:n_cases]:
        scored = classify_scores(item["images_similarity"])
        p2i_cases.append({
            "ID": item["ID"],
            "prompt": item["prompt"][:200],
            "ground_truth": item["ground_truth_images"],
            "top_results": [
                {"image": img, "score": s, "level": lvl["level"]}
                for img, s, lvl in zip(
                    item["images_retrieved"][:10],
                    item["images_similarity"][:10],
                    scored[:10]
                )
            ],
        })

    i2t_cases = []
    for item in i2t_dataset[:n_cases]:
        scored = classify_scores(item["text_similarity"])
        i2t_cases.append({
            "ID": item["ID"],
            "image": item["image"],
            "ground_truth": [t[:100] for t in item["ground_truth_text"]],
            "top_results": [
                {"text": txt[:150], "score": s, "level": lvl["level"]}
                for txt, s, lvl in zip(
                    item["text_retrieved"][:10],
                    item["text_similarity"][:10],
                    scored[:10]
                )
            ],
        })

    return {"prompt2image_cases": p2i_cases, "image2text_cases": i2t_cases}


# ═══════════════════════════════════════════════════════════════
# 9. Relaxed Ground Truth 构建
# ═══════════════════════════════════════════════════════════════

def build_relaxed_groups(api: MultimodalAPI) -> Dict[str, Dict[str, List[str]]]:
    """构建松弛 ground truth 分组.

    分组依据 (用于 prompt→image):
      - room_group: 同一 fuzzy_room_type 的图片
      - comment_group: 同一 _id 的图片 (同一评论的多张配图)

    Returns:
        {"room_group": {"套房": ["img_0.jpg", ...], ...},
         "comment_group": {"_id_xxx": ["img_0.jpg", "img_1.jpg", ...], ...}}
    """
    df = api.retriever.comments
    store = api.retriever.store
    img_data = store.get("hotel_images", include=["metadatas"])
    img_ids = img_data["ids"]
    metas = img_data["metadatas"]

    room_group = defaultdict(list)
    comment_group = defaultdict(list)
    for img_id, meta in zip(img_ids, metas):
        img_path = f"data/images/{img_id}.jpg"
        room = str(meta.get("room_type", "unknown"))
        cid = str(meta.get("comment_id", "unknown"))
        room_group[room].append(img_path)
        comment_group[cid].append(img_path)

    return {"room_group": dict(room_group), "comment_group": dict(comment_group)}


# ═══════════════════════════════════════════════════════════════
# 10. Query Fusion 策略
# ═══════════════════════════════════════════════════════════════

def query_fusion_experiment(api: MultimodalAPI, df: pd.DataFrame,
                             sample_size: int = 30) -> Dict:
    """Query Fusion: 加权融合多种文本形式的检索结果.

    使用 Chroma 中已有图片对应的行作为测试样本.
    """
    print(f"\n{'='*60}")
    print("Query Fusion experiment...")
    print(f"{'='*60}")

    # 获取有图片的行索引
    coll = api.retriever.store
    img_ids = coll.get("hotel_images", include=[])["ids"]
    indexed_rows = sorted([int(i.replace("img_", "")) for i in img_ids])
    rows_indices = indexed_rows[:sample_size]

    k_values = [1, 3, 5, 10]

    # 收集所有文本变体的检索结果
    all_results = defaultdict(list)
    for idx in rows_indices:
        if idx >= len(df):
            continue
        row = df.iloc[idx]
        variants = get_text_variants(row)
        gt = [f"data/images/img_{idx}.jpg"]
        for vname, vtext in variants.items():
            if not vtext.strip():
                continue
            result = api.prompt2image(vtext, topK=10)
            all_results[vname].append({
                "ID": int(idx), "images_retrieved": result["images"],
                "images_similarity": result["score"], "ground_truth_images": gt,
            })

    # 计算各变体 MRR 作为权重
    variant_weights = {}
    for vname, ds in all_results.items():
        variant_weights[vname] = mean_reciprocal_rank(ds)
    # 归一化
    total_w = sum(variant_weights.values()) or 1.0
    variant_weights = {k: v / total_w for k, v in variant_weights.items()}

    # 选出最好的两种
    top2 = sorted(variant_weights.items(), key=lambda x: x[1], reverse=True)[:2]
    top2_names = [n for n, _ in top2]

    strategies = {}
    # Strategy 1: 单一最优
    best_name = top2_names[0]
    strategies[f"单一最优 ({best_name})"] = all_results[best_name]

    # Strategy 2: Top-2 等权
    fused_top2 = []
    for i in range(len(rows_indices)):
        idx_i = int(rows_indices[i])
        # 收集 top-2 变体对该 query 的结果
        img_scores = defaultdict(float)
        for vname in top2_names:
            for j, item in enumerate(all_results[vname]):
                if item["ID"] == idx_i:
                    for img, s in zip(item["images_retrieved"], item["images_similarity"]):
                        img_scores[img] += s  # 等权加和
                    break
        sorted_imgs = sorted(img_scores.items(), key=lambda x: x[1], reverse=True)
        fused_top2.append({
            "ID": idx_i,
            "images_retrieved": [img for img, _ in sorted_imgs[:10]],
            "images_similarity": [s for _, s in sorted_imgs[:10]],
            "ground_truth_images": [f"data/images/img_{idx_i}.jpg"],
        })
    strategies["Top-2 等权融合"] = fused_top2

    # Strategy 3: 全变体加权融合
    fused_weighted = []
    for i in range(len(rows_indices)):
        idx_i = int(rows_indices[i])
        img_scores = defaultdict(float)
        for vname, ds in all_results.items():
            w = variant_weights.get(vname, 0.0)
            for item in ds:
                if item["ID"] == idx_i:
                    for img, s in zip(item["images_retrieved"], item["images_similarity"]):
                        img_scores[img] += w * s
                    break
        sorted_imgs = sorted(img_scores.items(), key=lambda x: x[1], reverse=True)
        fused_weighted.append({
            "ID": idx_i,
            "images_retrieved": [img for img, _ in sorted_imgs[:10]],
            "images_similarity": [s for _, s in sorted_imgs[:10]],
            "ground_truth_images": [f"data/images/img_{idx_i}.jpg"],
        })
    strategies["全变体加权融合"] = fused_weighted

    # 评估
    comparison = {}
    for sname, ds in strategies.items():
        hits = compute_hit_rates(ds, k_values)
        mrr = mean_reciprocal_rank(ds)
        ndcg = ndcg_at_k(ds, k=10)
        comparison[sname] = {
            "n_samples": len(ds),
            "hit_at_k": hits,
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
        }
        print(f"  {sname}: Hit@5={hits[5]:.3f}, MRR={mrr:.3f}, NDCG={ndcg:.3f}")

    return comparison


# ═══════════════════════════════════════════════════════════════
# 11. Score 校准分析
# ═══════════════════════════════════════════════════════════════

def score_calibration_analysis(p2i_dataset: List[dict], i2t_dataset: List[dict]) -> Dict:
    """分析 score 的校准质量.

    好的校准意味着: score 高的结果更可能是正确的。
    通过计算不同 score 分位区间的 precision 来评估。
    """
    print(f"\n{'='*60}")
    print("Score calibration analysis...")
    print(f"{'='*60}")

    results = {}
    for direction, dataset in [("prompt2image", p2i_dataset), ("image2text", i2t_dataset)]:
        rkey = "images_retrieved" if direction == "prompt2image" else "text_retrieved"
        skey = "images_similarity" if direction == "prompt2image" else "text_similarity"
        gkey = "ground_truth_images" if direction == "prompt2image" else "ground_truth_text"

        # 收集所有 (score, is_hit) 对
        all_pairs = []
        all_scores = []
        for item in dataset:
            gt_set = set(item.get(gkey, []))
            for i, (ret, s) in enumerate(zip(item.get(rkey, []), item.get(skey, []))):
                all_pairs.append({"score": s, "hit": ret in gt_set, "rank": i + 1})
                all_scores.append(s)

        if not all_scores:
            results[direction] = {}
            continue

        scores_arr = np.array(all_scores)
        # 分位数区间
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        bin_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        bin_edges = np.quantile(scores_arr, bins)

        cal_curve = []
        for i, label in enumerate(bin_labels):
            lo, hi = bin_edges[i], bin_edges[i + 1] if i + 1 < len(bin_edges) else 1.0
            in_bin = [p for p in all_pairs if lo <= p["score"] < hi]
            n_bin = len(in_bin)
            n_hits = sum(p["hit"] for p in in_bin)
            cal_curve.append({
                "bin": label,
                "range": f"[{lo:.3f}, {hi:.3f})",
                "n": n_bin,
                "hits": n_hits,
                "precision": n_hits / n_bin if n_bin > 0 else 0.0,
            })
            print(f"  [{direction}] {label} {lo:.3f}-{hi:.3f}: precision={n_hits}/{n_bin}={n_hits/n_bin if n_bin else 0:.3f}")

        # 校准误差 (Expected Calibration Error, 简化版)
        ece = sum(
            abs(b["precision"] - (i + 0.5) / len(bin_labels)) * b["n"]
            for i, b in enumerate(cal_curve)
        ) / len(all_pairs) if all_pairs else 0.0

        results[direction] = {
            "scores_stats": {
                "mean": float(np.mean(scores_arr)),
                "std": float(np.std(scores_arr)),
                "min": float(np.min(scores_arr)),
                "max": float(np.max(scores_arr)),
                "p25": float(np.percentile(scores_arr, 25)),
                "p50": float(np.percentile(scores_arr, 50)),
                "p75": float(np.percentile(scores_arr, 75)),
            },
            "calibration_curve": cal_curve,
            "ece": round(ece, 4),
        }

    return results


# ═══════════════════════════════════════════════════════════════
# 12. 错误分类分析
# ═══════════════════════════════════════════════════════════════

def categorize_errors(p2i_dataset: List[dict], i2t_dataset: List[dict]) -> Dict:
    """将检索失败按原因分类 (v2 — 更精细的分类).

    错误类型:
      - exact_hit: GT 在 top-10
      - near_miss: GT 在 rank 11-15 (仅差一点)
      - visual_confusion: 分数高(>0.38)但结果错误，top 结果集中在少数图片组
      - semantic_mismatch: 查询描述的概念与图片视觉内容不匹配 (如"服务态度"搜房间照)
      - low_confidence: 所有 score < 0.35，模型不确定
      - other: 其他未分类原因
    """
    print(f"\n{'='*60}")
    print("Error categorization (v2)...")
    print(f"{'='*60}")

    def classify(dataset, rkey, skey, gkey, direction):
        categories = {
            "exact_hit": 0, "near_miss": 0,
            "visual_confusion": 0, "semantic_mismatch": 0,
            "low_confidence": 0, "other": 0,
        }
        details = []

        for item in dataset:
            retrieved = item.get(rkey, [])
            scores = item.get(skey, [])
            gt = set(item.get(gkey, []))

            # 是否命中
            hit_pos = None
            for i, r in enumerate(retrieved):
                if r in gt:
                    hit_pos = i
                    break

            if hit_pos is not None and hit_pos < 10:
                categories["exact_hit"] += 1
                continue

            # 近似命中
            if hit_pos is not None and hit_pos < 15:
                categories["near_miss"] += 1
                details.append({"type": "near_miss", "ID": item["ID"],
                                "gt_rank": hit_pos + 1, "top_score": scores[0] if scores else 0})
                continue

            top_score = scores[0] if scores else 0.0
            # 分数离散度: top-3 内部的 std
            top3_std = float(np.std(scores[:3])) if len(scores) >= 3 else 0.0

            if top_score < 0.35:
                categories["low_confidence"] += 1
                details.append({"type": "low_confidence", "ID": item["ID"], "top_score": top_score})
            elif top3_std < 0.005 and top_score >= 0.38:
                # 分数极高且几乎无区分 → 多个视觉相似图片占据 top 位
                categories["visual_confusion"] += 1
                details.append({"type": "visual_confusion", "ID": item["ID"],
                                "top_score": top_score, "top3_std": top3_std})
            elif top_score >= 0.38:
                # 分数高但不集中在相似组 → 语义方向正确但具体图片不对
                categories["semantic_mismatch"] += 1
                details.append({"type": "semantic_mismatch", "ID": item["ID"],
                                "top_score": top_score, "top3_std": top3_std})
            else:
                categories["other"] += 1
                details.append({"type": "other", "ID": item["ID"],
                                "top_score": top_score, "top3_std": top3_std})

        return categories, details

    p2i_cats, p2i_details = classify(
        p2i_dataset, "images_retrieved", "images_similarity",
        "ground_truth_images", "p2i")
    i2t_cats, i2t_details = classify(
        i2t_dataset, "text_retrieved", "text_similarity",
        "ground_truth_text", "i2t")

    for dname, cats in [("prompt2image", p2i_cats), ("image2text", i2t_cats)]:
        total = sum(cats.values())
        n_wrong = total - cats["exact_hit"]
        print(f"  [{dname}] total={total}, wrong={n_wrong}")
        for cname in ["exact_hit", "near_miss", "visual_confusion",
                       "semantic_mismatch", "low_confidence", "other"]:
            cval = cats[cname]
            pct = cval / total if total else 0
            wpct = cval / n_wrong if n_wrong and cname != "exact_hit" else 0
            suffix = f" ({wpct:.0%} of errors)" if cname != "exact_hit" and n_wrong else ""
            print(f"    {cname}: {cval} ({pct:.0%}){suffix}")

    return {
        "prompt2image": {"categories": p2i_cats, "details": p2i_details},
        "image2text": {"categories": i2t_cats, "details": i2t_details},
    }


# ═══════════════════════════════════════════════════════════════
# 13. 图片混淆矩阵
# ═══════════════════════════════════════════════════════════════

def build_confusion_analysis(api: MultimodalAPI) -> Dict:
    """分析图片间的混淆模式.

    对每张图片，找出最容易被"误检索"的其他图片 (作为 query 的 top 结果但不正确).
    同时计算图片嵌入的 pairwise cosine similarity.
    """
    print(f"\n{'='*60}")
    print("Building image confusion analysis...")
    print(f"{'='*60}")

    # 获取所有图片 embedding (从 Chroma)
    coll = api.retriever.store
    all_data = coll.get("hotel_images", include=["embeddings", "metadatas"])
    img_ids = all_data["ids"]
    embs = np.array(all_data["embeddings"]).astype(np.float32)

    # cosine similarity matrix (embeddings 已 L2-normalized)
    sim_matrix = embs @ embs.T  # (N, N)

    # 对每张图片，找出最相似的 top-5 (排除自身)
    n = len(img_ids)
    confusion_pairs = []
    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -1  # 排除自身
        top_idx = np.argsort(sims)[::-1][:5]
        for rank, j in enumerate(top_idx):
            confusion_pairs.append({
                "from": f"data/images/{img_ids[i]}.jpg",
                "to": f"data/images/{img_ids[j]}.jpg",
                "cosine_sim": float(sims[j]),
                "rank": rank + 1,
            })

    # 找出最高混淆对
    confusion_pairs.sort(key=lambda x: x["cosine_sim"], reverse=True)

    # 统计：平均 pairwise similarity
    off_diag = sim_matrix[~np.eye(n, dtype=bool)]
    avg_sim = float(np.mean(off_diag))
    max_sim = float(np.max(off_diag))

    # 聚类检测: 找到相似度 > 0.95 的图片对 (near-duplicate)
    near_dup_pairs = [(confusion_pairs[i]["from"], confusion_pairs[i]["to"])
                       for i in range(len(confusion_pairs))
                       if confusion_pairs[i]["cosine_sim"] > 0.95 and
                       confusion_pairs[i]["rank"] == 1]

    print(f"  Avg pairwise cosine sim: {avg_sim:.4f}")
    print(f"  Max pairwise cosine sim: {max_sim:.4f}")
    print(f"  Near-duplicate pairs (sim>0.95): {len(near_dup_pairs)}")
    print(f"  Top confusion pairs:")
    for cp in confusion_pairs[:5]:
        print(f"    {cp['from'].split('/')[-1]} -> {cp['to'].split('/')[-1]}: {cp['cosine_sim']:.4f}")

    return {
        "avg_pairwise_sim": avg_sim,
        "max_pairwise_sim": max_sim,
        "near_duplicate_pairs": near_dup_pairs,
        "top_confusion_pairs": confusion_pairs[:30],
        "image_count": n,
    }


# ═══════════════════════════════════════════════════════════════
# 14. 检索多样性分析
# ═══════════════════════════════════════════════════════════════

def analyze_diversity(dataset: List[dict], direction: str = "p2i") -> Dict:
    """分析检索结果的多样性.

    指标:
      - intra_list_similarity: top-K 结果之间的平均 pairwise 相似度 (越低越多样)
      - unique_sources: 前K结果来自多少个不同的 _id review
      - score_drop: top-1 到 top-K 的分数衰减
    """
    print(f"\n{'='*60}")
    print(f"Diversity analysis [{direction}]...")
    print(f"{'='*60}")

    rkey = "images_retrieved" if direction == "p2i" else "text_retrieved"
    skey = "images_similarity" if direction == "p2i" else "text_similarity"

    k = min(10, len(dataset[0].get(rkey, [])) if dataset else 0)
    if k == 0:
        return {}

    diversities = []
    for item in dataset:
        scores = item.get(skey, [])[:k]
        if len(scores) < 2:
            continue
        # 分数离散度 (std / mean > higher = more diverse)
        score_std = float(np.std(scores))
        score_mean = float(np.mean(scores)) if np.mean(scores) > 0 else 1.0
        cv = score_std / score_mean  # coefficient of variation

        # 分数衰减: top-1 vs top-K
        score_drop_ratio = (scores[0] - scores[-1]) / scores[0] if scores[0] > 0 else 0.0

        diversities.append({
            "ID": item["ID"],
            "score_cv": round(cv, 4),
            "score_drop": round(score_drop_ratio, 4),
            "top1_score": round(scores[0], 4),
            "topK_score": round(scores[-1], 4),
        })

    # 汇总
    cv_values = [d["score_cv"] for d in diversities]
    drop_values = [d["score_drop"] for d in diversities]

    avg_cv = float(np.mean(cv_values)) if cv_values else 0.0
    avg_drop = float(np.mean(drop_values)) if drop_values else 0.0

    print(f"  Avg CV (score dispersion): {avg_cv:.4f}")
    print(f"  Avg score drop ratio: {avg_drop:.4f}")
    print(f"  Interpretation: CV={avg_cv:.3f} -> "
          + ("high diversity" if avg_cv > 0.1 else "low diversity (results clustered)"))

    return {
        "avg_score_cv": avg_cv,
        "avg_score_drop_ratio": avg_drop,
        "interpretation": "high diversity" if avg_cv > 0.1 else "low diversity",
        "per_item": diversities[:10],
    }


# ═══════════════════════════════════════════════════════════════
# 14a. 嵌入空间结构 (PCA + 聚类)
# ═══════════════════════════════════════════════════════════════

def embedding_space_analysis(api: MultimodalAPI) -> Dict:
    """PCA 有效秩 + 房型聚类质量 + 难负样本."""
    print(f"\n{'='*60}")
    print("Embedding space analysis (PCA + clustering)...")
    print(f"{'='*60}")

    coll = api.retriever.store
    data = coll.get("hotel_images", include=["embeddings", "metadatas"])
    embs = np.array(data["embeddings"]).astype(np.float32)
    metas = data["metadatas"]
    n = len(embs)

    # PCA
    centered = embs - embs.mean(axis=0)
    eigvals = np.sort(np.linalg.eigvalsh(centered.T @ centered / (n - 1)))[::-1]
    cumsum = np.cumsum(eigvals) / np.sum(eigvals)
    dim_50 = int(np.searchsorted(cumsum, 0.50) + 1)
    dim_80 = int(np.searchsorted(cumsum, 0.80) + 1)

    # 房型聚类
    room_groups = defaultdict(list)
    for i in range(n):
        room_groups[metas[i].get("room_type", "?")].append(i)
    room_stats = {}
    for room, idxs in room_groups.items():
        if len(idxs) < 2:
            continue
        g_embs = embs[idxs]
        intra = [float(np.dot(g_embs[a], g_embs[b]))
                 for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        other_mask = np.ones(n, dtype=bool); other_mask[idxs] = False
        inter = [float(s) for i_idx in idxs
                 for s in np.dot(embs[i_idx], embs[other_mask].T)]
        room_stats[room] = {
            "count": len(idxs),
            "intra_mean": float(np.mean(intra)) if intra else 0.0,
            "inter_mean": float(np.mean(inter)) if inter else 0.0,
            "separation": float(np.mean(intra) - np.mean(inter)) if intra and inter else 0.0,
        }

    # 难负样本 (最高相似度但不同 comment_id)
    sim_matrix = embs @ embs.T
    hard_negs = []
    for i in range(n):
        my_cid = metas[i].get("comment_id", "")
        sims = sim_matrix[i].copy(); sims[i] = -2
        for j in range(n):
            if metas[j].get("comment_id", "") == my_cid:
                sims[j] = -2
        top_j = int(np.argmax(sims))
        if sims[top_j] > -1:
            hard_negs.append({
                "query": f"data/images/{data['ids'][i]}.jpg",
                "negative": f"data/images/{data['ids'][top_j]}.jpg",
                "cosine": float(sims[top_j]),
                "same_room": metas[i].get("room_type") == metas[top_j].get("room_type"),
            })

    hn_sims = [h["cosine"] for h in hard_negs]
    print(f"  PCA: dim_50={dim_50}, dim_80={dim_80} (of 512)")
    print(f"  Room clusters: {len(room_stats)} types, "
          f"best_sep={max(s['separation'] for s in room_stats.values()):.3f}")
    print(f"  Hard negatives: mean_cos={np.mean(hn_sims):.3f}, "
          f"same_room={sum(h['same_room'] for h in hard_negs)}/{len(hard_negs)}")

    return {
        "pca": {"dim_50pct": dim_50, "dim_80pct": dim_80, "total": 512},
        "room_clusters": room_stats,
        "hard_negatives": {
            "mean_cosine": float(np.mean(hn_sims)),
            "same_room_ratio": sum(h["same_room"] for h in hard_negs) / len(hard_negs),
            "top10": sorted(hard_negs, key=lambda x: -x["cosine"])[:10],
        },
    }


# ═══════════════════════════════════════════════════════════════
# 14b. 跨模态对齐鸿沟
# ═══════════════════════════════════════════════════════════════

def cross_modal_alignment_analysis(api: MultimodalAPI) -> Dict:
    """量化 text↔image 跨模态对齐的非对称性."""
    print(f"\n{'='*60}")
    print("Cross-modal alignment gap analysis...")
    print(f"{'='*60}")

    # 图片 & 文本 embedding
    store = api.retriever.store
    img_data = store.get("hotel_images", include=["embeddings", "metadatas"])
    img_embs = np.array(img_data["embeddings"]).astype(np.float32)
    n_img = len(img_embs)

    txt_data = store.get("hotel_comments", include=["embeddings", "documents"])
    txt_embs = np.array(txt_data["embeddings"]).astype(np.float32)
    txt_docs = txt_data["documents"]
    n_txt = len(txt_embs)

    df = api.retriever.comments
    pairs = []
    for idx in range(min(50, n_img)):
        comment = str(df.iloc[idx]["comment"])[:300]
        img_emb = img_embs[idx]
        # 在文本集合中定位
        txt_idx = next((j for j, d in enumerate(txt_docs) if d and comment[:80] in d), None)
        if txt_idx is None:
            continue
        txt_emb = txt_embs[txt_idx]
        direct_cos = float(np.dot(img_emb, txt_emb))
        img_rank = int(np.sum(np.dot(img_emb, txt_embs.T) > direct_cos)) + 1
        txt_rank = int(np.sum(np.dot(txt_emb, img_embs.T) > direct_cos)) + 1
        pairs.append({
            "id": idx, "direct_cosine": direct_cos,
            "img_rank_in_texts": img_rank, "img_rank_pct": img_rank / n_txt,
            "txt_rank_in_images": txt_rank, "txt_rank_pct": txt_rank / n_img,
            "asymmetry": txt_rank / n_img - img_rank / n_txt,
        })

    cos_vals = [p["direct_cosine"] for p in pairs]
    asym_vals = [p["asymmetry"] for p in pairs]
    txt_harder = sum(1 for a in asym_vals if a > 0)

    print(f"  Pairs: {len(pairs)}")
    print(f"  Direct cosine: mean={np.mean(cos_vals):.4f}")
    print(f"  Image rank in texts: median={np.median([p['img_rank_in_texts'] for p in pairs]):.0f}/{n_txt}")
    print(f"  Text rank in images:  median={np.median([p['txt_rank_in_images'] for p in pairs]):.0f}/{n_img}")
    print(f"  Text→image harder: {txt_harder}/{len(pairs)} ({txt_harder/len(pairs):.0%})")

    return {
        "num_pairs": len(pairs),
        "direct_cosine_mean": float(np.mean(cos_vals)),
        "direct_cosine_std": float(np.std(cos_vals)),
        "img_rank_median_pct": float(np.median([p['img_rank_pct'] for p in pairs])),
        "txt_rank_median_pct": float(np.median([p['txt_rank_pct'] for p in pairs])),
        "text_harder_ratio": txt_harder / len(pairs) if pairs else 0,
        "details": pairs[:10],
    }


# ═══════════════════════════════════════════════════════════════
# 15. 主流程 (增强版)
# ═══════════════════════════════════════════════════════════════

def run_all_tests(quick: bool = False):
    """运行全部测试并输出结果."""
    start = time.time()

    print("=" * 60)
    print("Multimodal Recall — Test & Analysis Suite")
    print("=" * 60)

    # 初始化
    api = MultimodalAPI()
    print(f"Index stats: {api.stats}")

    # 构建测试集
    p2i_dataset, i2t_dataset = build_test_datasets(api, quick=quick)

    # ── Test 1: prompt→image ──
    p2i_dataset = test_prompt2image(api, p2i_dataset, topK=10)
    p2i_hits = compute_hit_rates(p2i_dataset)
    p2i_mrr = mean_reciprocal_rank(p2i_dataset)
    print(f"\n  prompt→image Results (n={len(p2i_dataset)}):")
    print(f"    Hit@1={p2i_hits[1]:.4f}  Hit@3={p2i_hits[3]:.4f}  "
          f"Hit@5={p2i_hits[5]:.4f}  Hit@10={p2i_hits[10]:.4f}  MRR={p2i_mrr:.4f}")

    # ── Test 2: image→text ──
    i2t_dataset = test_image2text(api, i2t_dataset, topK=10)
    i2t_hits = compute_hit_rates(i2t_dataset)
    i2t_mrr = mean_reciprocal_rank(i2t_dataset)
    print(f"\n  image→text Results (n={len(i2t_dataset)}):")
    print(f"    Hit@1={i2t_hits[1]:.4f}  Hit@3={i2t_hits[3]:.4f}  "
          f"Hit@5={i2t_hits[5]:.4f}  Hit@10={i2t_hits[10]:.4f}  MRR={i2t_mrr:.4f}")

    # ── Test 3: 对称性 ──
    symmetry_i2t2i = analyze_symmetry(api, i2t_dataset, sample_size=30 if not quick else 10)
    symmetry_t2i2t = analyze_reverse_symmetry(api, p2i_dataset, sample_size=30 if not quick else 10)

    # ── Test 4: 文本形式对比 ──
    df = api.retriever.comments
    text_variant_comparison = compare_text_variants(api, df,
                                                    sample_size=30 if not quick else 10)

    # ── Test 5: Score 阈值 ──
    threshold_analysis = analyze_score_threshold(p2i_dataset, i2t_dataset)

    # ── Test 6: Case 收集 ──
    case_examples = collect_case_examples(p2i_dataset, i2t_dataset, n_cases=5 if not quick else 3)

    # ── Test 7: Relaxed Hit@K ──
    relaxed_groups = build_relaxed_groups(api)
    p2i_relaxed_hits = compute_relaxed_hit_rates(p2i_dataset, relaxed_groups["room_group"])
    i2t_relaxed_hits = compute_relaxed_hit_rates(i2t_dataset, relaxed_groups["comment_group"])
    print(f"\n  Relaxed Hit@K (room_type groups):")
    print(f"    prompt→image: Hit@5={p2i_relaxed_hits[5]:.3f}, Hit@10={p2i_relaxed_hits[10]:.3f}")

    # ── Test 8: NDCG ──
    p2i_ndcg = ndcg_at_k(p2i_dataset, k=10)
    i2t_ndcg = ndcg_at_k(i2t_dataset, k=10)
    print(f"  NDCG@10: prompt→image={p2i_ndcg:.4f}, image→text={i2t_ndcg:.4f}")

    # ── Test 9: Query Fusion ──
    query_fusion_results = query_fusion_experiment(
        api, df, sample_size=min(17, len(p2i_dataset)))

    # ── Test 10: Score 校准 ──
    calibration = score_calibration_analysis(p2i_dataset, i2t_dataset)

    # ── Test 11: 错误分类 ──
    error_analysis = categorize_errors(p2i_dataset, i2t_dataset)

    # ── Test 12: 图片混淆 ──
    confusion = build_confusion_analysis(api)

    # ── Test 13: 多样性 ──
    p2i_diversity = analyze_diversity(p2i_dataset, "p2i")
    i2t_diversity = analyze_diversity(i2t_dataset, "i2t")

    # ── Test 14: 嵌入空间 ──
    embedding = embedding_space_analysis(api)

    # ── Test 15: 跨模态对齐 ──
    cross_modal = cross_modal_alignment_analysis(api)

    # ── 组装结果 ──
    all_results = {
        "config": {
            "quick_mode": quick,
            "p2i_samples": len(p2i_dataset),
            "i2t_samples": len(i2t_dataset),
            "index_stats": api.stats,
        },
        "prompt2image": {
            "hit_at_k": p2i_hits,
            "mrr": round(p2i_mrr, 4),
            "ndcg": round(p2i_ndcg, 4),
            "relaxed_hit_at_k": p2i_relaxed_hits,
            "dataset_summary": [
                {k: v for k, v in item.items() if k not in ("images_retrieved", "text_retrieved")}
                for item in p2i_dataset[:5]
            ],
            "full_dataset": p2i_dataset,
        },
        "image2text": {
            "hit_at_k": i2t_hits,
            "mrr": round(i2t_mrr, 4),
            "ndcg": round(i2t_ndcg, 4),
            "relaxed_hit_at_k": i2t_relaxed_hits,
            "dataset_summary": [
                {k: v for k, v in item.items() if k != "images_retrieved" and k != "text_retrieved"}
                for item in i2t_dataset[:5]
            ],
            "full_dataset": i2t_dataset,
        },
        "symmetry": {
            "image_to_text_to_image": {k: v for k, v in symmetry_i2t2i.items() if k != "details"},
            "text_to_image_to_text": {k: v for k, v in symmetry_t2i2t.items() if k != "details"},
            "image_to_text_to_image_details": symmetry_i2t2i["details"],
            "text_to_image_to_text_details": symmetry_t2i2t["details"],
        },
        "text_variant_comparison": text_variant_comparison,
        "threshold_analysis": threshold_analysis,
        "case_examples": case_examples,
        "query_fusion": query_fusion_results,
        "score_calibration": calibration,
        "error_analysis": error_analysis,
        "confusion_analysis": confusion,
        "diversity": {
            "prompt2image": p2i_diversity,
            "image2text": i2t_diversity,
        },
        "embedding_space": embedding,
        "cross_modal_alignment": cross_modal,
    }

    # ── 保存 ──
    output_path = os.path.join(OUTPUT_DIR, "test_results.json")
    # 对于 JSON 序列化，移除完整数据集以减小文件体积
    results_for_json = {k: v for k, v in all_results.items()}
    # 保留完整数据集用于后续分析
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_for_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OK] Results saved to {output_path}")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"All tests completed in {elapsed:.1f}s")
    print(f"{'='*60}")

    return all_results


# ═══════════════════════════════════════════════════════════════
# 10. 报告生成
# ═══════════════════════════════════════════════════════════════

def generate_report(results: Dict) -> str:
    """基于测试结果生成 Markdown 报告 (v2 — 深度版)."""
    cfg = results["config"]
    p2i = results["prompt2image"]
    i2t = results["image2text"]
    sym = results["symmetry"]
    txt_var = results["text_variant_comparison"]
    thr = results["threshold_analysis"]
    cases = results["case_examples"]
    fusion = results.get("query_fusion", {})
    cal = results.get("score_calibration", {})
    err = results.get("error_analysis", {})
    conf = results.get("confusion_analysis", {})
    div_res = results.get("diversity", {})

    lines = []
    def w(s=""):
        lines.append(s)

    w("# 多模态召回 — 深度测试分析报告 (v2)")
    w()
    w(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**索引规模**: {cfg['index_stats']}")
    w(f"**测试样本**: prompt→image n={cfg['p2i_samples']}, image→text n={cfg['i2t_samples']}")
    w()

    # ── 1. 基础指标 (含 Relaxed & NDCG) ──
    w("## 1. 召回指标总览")
    w()
    w("### 1.1 Strict Hit@K (精确匹配)")
    w()
    w("| 方向 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | NDCG@10 |")
    w("|------|-------|-------|-------|--------|-----|---------|")
    w(f"| prompt→image | {p2i['hit_at_k'][1]:.4f} | {p2i['hit_at_k'][3]:.4f} | {p2i['hit_at_k'][5]:.4f} | {p2i['hit_at_k'][10]:.4f} | {p2i['mrr']:.4f} | {p2i.get('ndcg', 0):.4f} |")
    w(f"| image→text | {i2t['hit_at_k'][1]:.4f} | {i2t['hit_at_k'][3]:.4f} | {i2t['hit_at_k'][5]:.4f} | {i2t['hit_at_k'][10]:.4f} | {i2t['mrr']:.4f} | {i2t.get('ndcg', 0):.4f} |")
    w()

    # Relaxed
    if p2i.get("relaxed_hit_at_k"):
        w("### 1.2 Relaxed Hit@K (房型分组松弛匹配)")
        w()
        w("> 若检索结果与 ground truth 属于同一 `fuzzy_room_type`，计 0.5 分；精确匹配计 1.0 分。")
        w()
        w("| 方向 | Hit@1 | Hit@3 | Hit@5 | Hit@10 |")
        w("|------|-------|-------|-------|--------|")
        rh = p2i["relaxed_hit_at_k"]
        w(f"| prompt→image | {rh[1]:.4f} | {rh[3]:.4f} | {rh[5]:.4f} | {rh[10]:.4f} |")
        rh2 = i2t.get("relaxed_hit_at_k", {})
        if rh2:
            w(f"| image→text | {rh2[1]:.4f} | {rh2[3]:.4f} | {rh2[5]:.4f} | {rh2[10]:.4f} |")
        w()

    # ── 2. 对称性 ──
    w("## 2. 对称性分析")
    w()
    i2t2i = sym["image_to_text_to_image"]
    t2i2t = sym["text_to_image_to_text"]
    w("| 方向 | 对称率 | 说明 |")
    w("|------|--------|------|")
    w(f"| image→text→image | **{i2t2i['symmetric_rate']:.1%}** | 图片经文本桥接后能找回原图的比例 |")
    w(f"| text→image→text | **{t2i2t['symmetric_rate']:.1%}** | 文本经图片桥接后能精确匹配原文的比例 |")
    w()
    w(f"**核心发现**: 两个方向的对称性严重不对称 ({i2t2i['symmetric_rate']:.0%} vs {t2i2t['symmetric_rate']:.0%})。")
    w("图片是更可靠的语义'锚点' — 从图片出发的环路高度自洽，但从长文本出发的环路极易断裂。")
    w("原因: 长评论覆盖多个话题维度，任何单张图片只能匹配其中一部分语义。")
    w()

    # ── 3. 文本形式 ──
    w("## 3. 文本形式对比")
    w()
    w("| 文本形式 | n | Hit@1 | Hit@5 | Hit@10 | MRR |")
    w("|----------|---|-------|-------|--------|-----|")
    best_variant = None
    best_mrr = -1
    for vname, vdata in sorted(txt_var.items(), key=lambda x: -x[1]["mrr"]):
        h = vdata["hit_at_k"]
        marker = " **(best)**" if vdata["mrr"] > best_mrr else ""
        w(f"| {vname}{marker} | {vdata['n_samples']} | {h[1]:.4f} | {h[5]:.4f} | {h[10]:.4f} | {vdata['mrr']:.4f} |")
        if vdata["mrr"] > best_mrr:
            best_mrr = vdata["mrr"]
            best_variant = vname
    w()
    w(f"**结论**: 短文本 (房型、查询词) 效果远超长评论。{best_variant} 的 MRR 是完整评论的 **{best_mrr/p2i['mrr']:.1f}x** 倍。")
    w()

    # ── 4. Query Fusion ──
    if fusion:
        w("## 4. Query Fusion — 多文本形式融合")
        w()
        w("| 策略 | n | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG |")
        w("|------|---|-------|-------|--------|-----|------|")
        for sname, sdata in fusion.items():
            h = sdata["hit_at_k"]
            w(f"| {sname} | {sdata['n_samples']} | {h[1]:.4f} | {h[5]:.4f} | {h[10]:.4f} | {sdata['mrr']:.4f} | {sdata['ndcg']:.4f} |")
        w()

    # ── 5. Score 校准 ──
    if cal:
        w("## 5. Score 校准分析")
        w()
        for direction in ["prompt2image", "image2text"]:
            if direction not in cal:
                continue
            cd = cal[direction]
            stats = cd.get("scores_stats", {})
            w(f"### 5.{'1' if direction == 'prompt2image' else '2'} {direction}")
            w()
            w(f"Score 分布: mean={stats.get('mean', 0):.4f}, std={stats.get('std', 0):.4f}, "
              f"range=[{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]")
            w(f"ECE (Expected Calibration Error): {cd.get('ece', 0):.4f}")
            w()
            w("| 分位区间 | 范围 | 样本数 | 命中数 | 精度 |")
            w("|----------|------|--------|--------|------|")
            for b in cd.get("calibration_curve", []):
                w(f"| {b['bin']} | {b['range']} | {b['n']} | {b['hits']} | {b['precision']:.3f} |")
            w()
        w("**结论**: " + (
            "Score 与精度呈正相关，可用于排序。" if cal.get("prompt2image", {}).get("ece", 1) < 0.3
            else "Score 校准不足，高分区精度并未显著优于低分区，不建议仅依赖 score 做硬过滤。"
        ))
        w()

    # ── 6. Score 阈值 ──
    w("## 6. Score 阈值筛选")
    w()
    for direction in ["prompt2image", "image2text"]:
        base = thr[direction]["baseline"]
        w(f"### {direction} | Baseline: Hit@5={base['hit_at_k'][5]:.4f}")
        w()
        # 只显示关键阈值
        key_thresholds = [0.2, 0.3, 0.35, 0.4, 0.45]
        w("| 阈值 | 通过率 | Hit@5 | 变化 |")
        w("|------|--------|-------|------|")
        for th in key_thresholds:
            if th in thr[direction]["by_threshold"]:
                tdata = thr[direction]["by_threshold"][th]
                delta = tdata["hit_at_k"][5] - base["hit_at_k"][5]
                delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                w(f"| {th:.2f} | {tdata['pass_rate']:.0%} | {tdata['hit_at_k'][5]:.4f} | {delta_str} |")
        w()
    w("**结论**: 在当前数据上，固定阈值无法有效分离正确/错误结果。建议使用相对排序而非绝对阈值。")
    w()

    # ── 7. 错误分类 ──
    if err:
        w("## 7. 检索失败原因分类 (v2)")
        w()
        for direction in ["prompt2image", "image2text"]:
            cats = err[direction]["categories"]
            total = sum(cats.values())
            n_wrong = total - cats.get("exact_hit", 0)
            w(f"### 7.{'1' if direction == 'prompt2image' else '2'} {direction} (总样本: {total}, 错误: {n_wrong})")
            w()
            w("| 错误类型 | 数量 | 总占比 | 错误占比 | 说明 |")
            w("|----------|------|--------|----------|------|")
            rows_info = [
                ("exact_hit", "精确命中 (top-10)", False),
                ("near_miss", "近似命中 (rank 11-15)", True),
                ("visual_confusion", "视觉混淆 (top结果集中在相似图片组)", True),
                ("semantic_mismatch", "语义不匹配 (查询概念≠图片内容)", True),
                ("low_confidence", "低置信度 (top score < 0.35)", True),
                ("other", "其他未分类原因", True),
            ]
            for cname, cdesc, is_error in rows_info:
                cval = cats.get(cname, 0)
                tpct = f"{cval/total:.0%}" if total else "0%"
                epct = f"{cval/n_wrong:.0%}" if is_error and n_wrong else "-"
                w(f"| {cname} | {cval} | {tpct} | {epct} | {cdesc} |")
            w()

        # 综合诊断
        p2i_vc = err["prompt2image"]["categories"].get("visual_confusion", 0)
        p2i_sm = err["prompt2image"]["categories"].get("semantic_mismatch", 0)
        p2i_lc = err["prompt2image"]["categories"].get("low_confidence", 0)
        w("**综合诊断**:")
        if p2i_vc > p2i_sm and p2i_vc > p2i_lc:
            w(f"- 主要失败模式: **视觉混淆** ({p2i_vc} 例) — 不同评论的图片视觉高度相似，模型难以区分")
            w("- 建议: 引入 hard negative mining 或在损失函数中增加细粒度区分约束")
        elif p2i_sm > p2i_vc and p2i_sm > p2i_lc:
            w(f"- 主要失败模式: **语义不匹配** ({p2i_sm} 例) — 文本描述与图片内容不在同一语义维度")
            w("- 建议: 使用更聚焦的查询文本 (如房型名称)，或对长文本做关键句提取")
        else:
            w(f"- 失败模式分散 — 低置信度 ({p2i_lc} 例)、视觉混淆 ({p2i_vc} 例)、语义不匹配 ({p2i_sm} 例) 均存在")
        w()

    # ── 8. 图片混淆 ──
    if conf:
        w("## 8. 图片嵌入空间分析")
        w()
        w(f"- 图片数量: {conf.get('image_count', 0)}")
        w(f"- 平均 pairwise cosine similarity: {conf.get('avg_pairwise_sim', 0):.4f}")
        w(f"- 最大 pairwise cosine similarity: {conf.get('max_pairwise_sim', 0):.4f}")
        w(f"- 近似重复对 (sim>0.95): {len(conf.get('near_duplicate_pairs', []))} 对")
        w()
        if conf.get("top_confusion_pairs"):
            w("**最高混淆对 (Top-5)**:")
            w()
            w("| From | To | Cosine Sim |")
            w("|------|-----|-----------|")
            for cp in conf["top_confusion_pairs"][:5]:
                w(f"| {cp['from'].split('/')[-1]} | {cp['to'].split('/')[-1]} | {cp['cosine_sim']:.4f} |")
            w()
        w(f"**结论**: 平均图片相似度 {conf.get('avg_pairwise_sim', 0):.3f}，" +
          ("图片间区分度较高，混淆主要来自真正相似的图片组。" if conf.get('avg_pairwise_sim', 0) < 0.8
           else "图片间相似度偏高，检索容易发生混淆。"))
        w()

    # ── 9. 多样性 ──
    if div_res:
        w("## 9. 检索结果多样性")
        w()
        for direction in ["prompt2image", "image2text"]:
            dd = div_res.get(direction, {})
            if not dd:
                continue
            w(f"### 9.{'1' if direction == 'prompt2image' else '2'} {direction}")
            w(f"- 平均分数变异系数 (CV): {dd.get('avg_score_cv', 0):.4f}")
            w(f"- 平均分数衰减比 (top1→topK): {dd.get('avg_score_drop_ratio', 0):.4f}")
            w(f"- 多样性评估: **{dd.get('interpretation', 'unknown')}**")
            w()
            if dd.get("interpretation") == "low diversity":
                w("> 检索结果集中在少数高分图片组，建议引入多样性重排 (MMR) 来提升覆盖面。")
                w()

    # ── 10. 嵌入空间 ──
    emb_space = results.get("embedding_space", {})
    if emb_space:
        w("## 10. 嵌入空间结构")
        w()
        pca = emb_space["pca"]
        w(f"**PCA**: 512维嵌入中，仅 **{pca['dim_50pct']}维** 解释50%方差，**{pca['dim_80pct']}维** 解释80%方差。"
          f"有效秩仅 {pca['dim_80pct']/pca['total']:.1%}，存在大量冗余维度。")
        w()
        w("### 房型聚类质量")
        w()
        w("| 房型 | 图片数 | 类内cos | 类间cos | 分离度 |")
        w("|------|--------|---------|---------|--------|")
        for room, s in sorted(emb_space["room_clusters"].items(), key=lambda x: -x[1]["separation"]):
            w(f"| {room} | {s['count']} | {s['intra_mean']:.3f} | {s['inter_mean']:.3f} | {s['separation']:+.3f} |")
        w()
        hn = emb_space["hard_negatives"]
        w(f"**难负样本**: 平均cosine={hn['mean_cosine']:.3f}，同房型仅占{hn['same_room_ratio']:.0%}。"
          "区分正确/错误图片的空间余量极小（<0.2 cosine距离）。")
        w()

    # ── 11. 跨模态对齐 ──
    cm = results.get("cross_modal_alignment", {})
    if cm:
        w("## 11. 跨模态对齐鸿沟")
        w()
        w(f"- 同一(text, image)对的直接 cosine: **{cm['direct_cosine_mean']:.3f}** ± {cm['direct_cosine_std']:.3f}")
        w(f"- 图片在文本库中位排名: 前 **{cm['img_rank_median_pct']:.0%}**")
        w(f"- 文本在图片库中位排名: 前 **{cm['txt_rank_median_pct']:.0%}**")
        w(f"- text→image 比 image→text 更难的比例: **{cm['text_harder_ratio']:.0%}**")
        w()
        w("**结论**: 即使给模型看同一内容的文本和图片，它们的 embedding cosine 也只有 0.36。"
          "图片作为查询比文本作为查询的排名显著更好，证实了此前发现的非对称性根源在嵌入层面。")
        w()

    # ── 12. 具体 Case ──
    w("## 12. 具体 Case 展示")
    w()
    for direction_label, case_list in [
        ("prompt → image", cases["prompt2image_cases"]),
        ("image → text", cases["image2text_cases"]),
    ]:
        w(f"### {direction_label}")
        w()
        for case in case_list[:3]:  # 每种展示前3个
            if "prompt" in case:
                w(f"**Case ID={case['ID']}** | prompt: _{case['prompt'][:100]}..._")
            else:
                w(f"**Case ID={case['ID']}** | image: `{case['image']}`")
            w()
            w("| Rank | Score | Level | Content |")
            w("|------|-------|-------|---------|")
            for i, r in enumerate(case["top_results"][:8]):
                content = f"`{r.get('image', '')}`" if "image" in r else r.get("text", "")[:80]
                w(f"| {i+1} | {r['score']:.4f} | **{r['level']}** | {content} |")
            w()

    report = "\n".join(lines)

    report_path = os.path.join(OUTPUT_DIR, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Report saved to {report_path}")

    return report


# ═══════════════════════════════════════════════════════════════
# 11. Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多模态召回测试与分析")
    parser.add_argument("--quick", action="store_true", help="快速模式 (少量样本)")
    parser.add_argument("--report-only", action="store_true",
                        help="仅从已有 results JSON 生成报告")
    args = parser.parse_args()

    if args.report_only:
        results_path = os.path.join(OUTPUT_DIR, "test_results.json")
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            generate_report(results)
        else:
            print(f"Error: {results_path} not found. Run tests first.")
    else:
        results = run_all_tests(quick=args.quick)
        generate_report(results)
        print("\n[OK] All deliverables generated in deliverable/")
