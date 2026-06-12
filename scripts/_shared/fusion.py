"""
结果融合工具 — RRF (Reciprocal Rank Fusion) + HyDE 生成.

从 scripts/dyx/lib.py 提取重构。

用法:
    from _shared.fusion import rrf_fuse, hyde_generate

    # RRF 融合多路检索结果
    fused = rrf_fuse([vector_results, bm25_results], k=60)

    # HyDE 生成假设文档（需 LLM）
    docs = hyde_generate(query, chat_fn)
"""

from typing import Callable, Dict, List, Optional


def rrf_fuse(
    retrieval_lists: List[List[Dict]],
    k: int = 60,
    topk: int = 10,
) -> List[Dict]:
    """RRF (Reciprocal Rank Fusion) 融合多路检索结果.

    Args:
        retrieval_lists: 每路检索结果列表, 每路按 rank 升序排列.
        k: RRF 常数 (默认 60).
        topk: 返回 top-k 结果.

    Returns:
        融合后的结果列表, 每项包含原字段 + fused_score.
    """
    rrf_scores: Dict[str, float] = {}
    all_items: Dict[str, Dict] = {}

    for retrieval_list in retrieval_lists:
        for item in retrieval_list:
            doc_id = item.get('doc_id', '')
            rank = item.get('rank', 0)
            if not doc_id:
                continue
            score = 1.0 / (k + rank)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                all_items[doc_id] = dict(item)
            rrf_scores[doc_id] += score

    # 按 RRF 分数降序排列
    fused = []
    for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(all_items[doc_id])
        item['fused_score'] = round(score, 6)
        fused.append(item)
        if len(fused) >= topk:
            break
    return fused


def hyde_generate(
    query: str,
    chat_fn: Callable,
    max_attempts: int = 2,
) -> List[str]:
    """HyDE: 用 LLM 生成假设文档, 用于增强检索.

    Args:
        query: 用户查询.
        chat_fn: LLM 调用函数 chat(messages) -> str.
        max_attempts: 最大生成数量.

    Returns:
        生成的假设文档列表.
    """
    prompt = (
        f"用户搜索: {query}\
\
"
        f"请用一段真实的酒店评论口吻, 写出可能包含相关信息的评论内容.\
"
        f"只输出评论本身, 不要解释."
    )
    docs = []
    for _ in range(max_attempts):
        try:
            result = chat_fn([
                {'role': 'system', 'content': '你是酒店评论助手.'},
                {'role': 'user', 'content': prompt},
            ])
            if result and result.strip():
                docs.append(result.strip())
        except Exception:
            pass
    return docs


def merge_strategy_results(
    multimodal_result: Optional[Dict] = None,
    rag_result: Optional[Dict] = None,
    bm25_result: Optional[Dict] = None,
) -> Dict:
    """合并多策略返回结果为统一格式.

    用于 _shared/router.py 集成多路召回.
    """
    merged = {'strategies': {}}

    if multimodal_result:
        merged['strategies']['multimodal'] = multimodal_result
    if rag_result:
        merged['strategies']['rag'] = rag_result
    if bm25_result:
        merged['strategies']['bm25_enhanced'] = bm25_result

    return merged
