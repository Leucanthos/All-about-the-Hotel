"""
dyx 核心算法引擎 — 从 lib.py / lib_syh.py 提取重构.

包含:
  1. BM25 倒排索引 (迁移至 _shared/bm25.py, 此处 re-export)
  2. 细粒度意图分类
  3. 查询扩展
  4. RRF 多路融合 (迁移至 _shared/fusion.py, 此处 re-export)
  5. 双路/三路召回编排
  6. 结果重排序
"""

from _shared.bm25 import InvertedIndex
from _shared.fusion import rrf_fuse, hyde_generate

# ── 意图分类 ──

INTENT_CATEGORIES = {
    'facility':       ['设施', '设备', '装修', '房间', '游泳池', '健身房', '花园',
                        '停车场', '电梯', '空调', '热水', '无线', 'wifi', '网络'],
    'price':          ['价格', '性价比', '费用', '划算', '值得', '贵', '便宜',
                        '值', '优惠', '折扣', '房价'],
    'location':       ['位置', '交通', '周边', '地铁', '商圈', '距离', '出行',
                        '方便', '市中心', '景区'],
    'service':        ['服务', '前台', '态度', '专业', '热情', '效率', '办理',
                        '入住', '退房', '客服', '管家'],
    'food':           ['早餐', '餐饮', '餐厅', '美食', '食品', '自助餐',
                        '菜品', '口味', '好吃'],
    'cleanliness':    ['卫生', '干净', '清洁', '整洁', '脏', '异味', '霉'],
    'quiet':          ['安静', '噪音', '隔音', '吵', '嘈杂', '睡眠'],
    'child_friendly': ['亲子', '儿童', '小孩', '孩子', '家庭', '乐园'],
    'elder_friendly': ['老人', '长辈', '轮椅', '无障碍'],
    'overall':        ['推荐', '体验', '感受', '总结', '评价', '怎么样', '如何'],
}


def detect_intent(query: str) -> dict:
    """关键词检测多维意图."""
    scores = {}
    for cat, keywords in INTENT_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in query)
        if score > 0:
            scores[cat] = score
    if not scores:
        return {'primary': 'general', 'categories': [], 'confidence': 0.0}
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    total = sum(scores.values())
    return {
        'primary': sorted_cats[0][0],
        'categories': [c for c, _ in sorted_cats],
        'confidence': round(sorted_cats[0][1] / total, 4) if total > 0 else 0,
    }


# ── 查询扩展 ──

def expand_query_llm(query: str, chat_fn=None) -> list:
    """LLM 查询扩展."""
    if chat_fn is None:
        return [query]
    try:
        prompt = (
            f"用户想搜索酒店评论: {query}\
\
"
            f"请生成 1~2 个相关的搜索意图, 帮助更全面地检索.\
"
            f"每行一个, 尽量简洁, 不要编号."
        )
        result = chat_fn([
            {'role': 'system', 'content': '你是酒店搜索助手.'},
            {'role': 'user', 'content': prompt},
        ])
        if result:
            exps = [line.strip().strip('-').strip()
                   for line in result.strip().split('\
')
                   if line.strip() and len(line.strip()) > 2]
            seen = {query}
            queries = [query]
            for e in exps:
                if e not in seen:
                    queries.append(e)
                    seen.add(e)
            return queries[:3]
    except Exception:
        pass
    return [query]


# ── 兼容原 lib_syh.py 的检索编排 ──

def build_retrieval_pipeline(
    query: str,
    inverted_index: InvertedIndex,
    vector_search_fn=None,
    reverse_query_fn=None,
    top_k: int = 10,
) -> dict:
    """编排多路检索: BM25 + 向量 + 反向 Query.

    模拟 lib_syh.py 中 RAGPipeline.run() 的核心逻辑.

    Returns:
        {query, intent, bm25_results, vector_results, reverse_query_results,
         fused_results (RRF)}
    """
    intent = detect_intent(query)
    result = {
        'query': query,
        'intent': intent,
        'bm25_results': [],
        'vector_results': [],
        'reverse_query_results': [],
        'fused_results': [],
    }

    # BM25
    bm25_raw = inverted_index.search(query, topk=top_k * 2)
    for rank, (doc_id, score) in enumerate(bm25_raw, 1):
        result['bm25_results'].append({
            'rank': rank, 'method': 'bm25',
            'doc_id': doc_id, 'score': round(score, 4),
            'comment': inverted_index.documents.get(doc_id, '')[:300],
        })

    # 向量检索
    if vector_search_fn:
        try:
            vec_items = vector_search_fn(query, top_k * 2)
            for i, item in enumerate(vec_items):
                if isinstance(item, dict):
                    item['rank'] = i + 1
                    item['method'] = 'vector'
                    result['vector_results'].append(item)
        except Exception:
            pass

    # 反向 Query 检索
    if reverse_query_fn:
        try:
            rev_items = reverse_query_fn(query, top_k * 2)
            for i, item in enumerate(rev_items):
                if isinstance(item, dict):
                    item['rank'] = i + 1
                    item['method'] = 'reverse_query'
                    result['reverse_query_results'].append(item)
        except Exception:
            pass

    # RRF 融合
    lists_to_fuse = [result['bm25_results']]
    if result['vector_results']:
        lists_to_fuse.append(result['vector_results'])
    if result['reverse_query_results']:
        lists_to_fuse.append(result['reverse_query_results'])

    if len(lists_to_fuse) > 1:
        result['fused_results'] = rrf_fuse(lists_to_fuse, topk=top_k)
    else:
        result['fused_results'] = result['bm25_results'][:top_k]

    return result
