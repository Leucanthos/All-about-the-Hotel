"""Layer 2 LLM Router — intelligent strategy selection (SU Leuca + SUN Yuhao).

Modes:
  1. LLM routing: DeepSeek API analyzes query intent -> selects best strategy
  2. Keyword fallback: pattern matching when no API key available

Env vars (priority: system env > .env file):
  DEEPSEEK_API_KEY    DeepSeek API key
  LLM_API_KEY         Generic API key (OpenAI-compatible)
  LLM_API_BASE        API base URL (default: https://api.deepseek.com)
  LLM_MODEL           Model name (default: deepseek-chat)

Setup: https://platform.deepseek.com/api_keys | https://platform.deepseek.com/top_up
"""

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import List, Optional


# -- .env loader (no extra dependencies) --

def _load_dotenv():
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val

_load_dotenv()

DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

SETUP_HINT = """[LLM] No API key found. Get one at https://platform.deepseek.com/api_keys (Y=1 minimum).
[LLM] Put it in .env file:  DEEPSEEK_API_KEY=sk-xxx
[LLM] Or set env var:  set DEEPSEEK_API_KEY=sk-xxx
[LLM] Falling back to keyword routing."""


# -- Keyword fallback --

_QA_KW = {
    "how", "suitable", "recommend", "good", "bad", "review", "experience",
    "service", "food", "breakfast", "location", "price", "worth",
    "noise", "clean", "comfortable", "staff", "check", "child", "elder",
    "- what", "- how", "- is", "- are", "- can", "- should", "- do",
}

_VISUAL_KW = {
    "suite", "room", "bed", "pool", "garden", "lobby", "exterior",
    "view", "decoration", "luxury", "window", "corridor", "bathroom",
}


def classify_intent(query: str) -> str:
    """Keyword fallback: qa | visual | mixed."""
    q_lower = query.lower()
    is_qa = any(kw in q_lower for kw in _QA_KW)
    is_visual = any(kw in q_lower for kw in _VISUAL_KW)
    if is_qa and is_visual:
        return "mixed"
    if is_qa:
        return "qa"
    if is_visual:
        return "visual"
    return "mixed"


# -- LLM routing --

SYSTEM_PROMPT = """You are an intent router for a hotel review system. Analyze user queries and decide which retrieval strategy to use.

Available strategies:
- "multimodal": User wants to SEE hotel photos (facilities, rooms, exterior, pool, garden, etc.)
- "rag": User asks about service quality, dining, location, suitability, experience, recommendations
- "all": Query touches both visual AND experiential aspects

Return ONLY valid JSON: {"intent": "<multimodal|rag|all>", "reason": "<one sentence>"}"""


def classify_intent_llm(query: str) -> Optional[dict]:
    """LLM 意图分类 — 本地模型 > DeepSeek API > None."""
    from _shared.llm import classify_json
    result = classify_json(SYSTEM_PROMPT, query)
    if result:
        intent = result.get("intent", "mixed")
        if intent not in ("multimodal", "rag", "all"):
            intent = "mixed"
        return {"intent": intent, "reason": result.get("reason", ""),
                "model": "local" if _is_local_active() else (os.getenv("LLM_MODEL") or DEEPSEEK_MODEL),
                "router": "llm"}
    return None


def _is_local_active() -> bool:
    try:
        from _shared.llm import is_local_available
        return is_local_available()
    except Exception:
        return False


def _intent_to_strategies(intent: str) -> list[str]:
    if intent == "multimodal":
        return ["multimodal"]
    if intent == "rag":
        return ["rag"]
    return ["multimodal", "rag"]


# -- Engines (使用共享缓存 _shared/cache, 避免重复加载) --

def _run_multimodal(query: str, top_k: int = 10) -> Optional[dict]:
    """多模态检索 — Leuca 贡献 (共享缓存)."""
    try:
        from _shared.cache import cache
        retriever = cache.get_retriever()
        r = retriever.search_images(query, top_k=top_k)
        return {"images": [item.get("comment", "") for item in r],
                "scores": [round(item.get("similarity", 0), 4) for item in r],
                "image_ids": [item.get("image_id", "") for item in r],
                "method": "Chinese-CLIP", "latency_ms": 0}
    except Exception as e:
        return {"error": str(e)}


def _run_rag(query: str, top_k: int = 8) -> Optional[dict]:
    """Agentic RAG — Yuhao 贡献 (共享缓存)."""
    try:
        from yuhao.agentic.api import answer_question
        has_key = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY"))
        output = answer_question(query, top_k=top_k, use_llm=has_key)
        result = output["result"]
        return {"answer": output.get("answer", "")[:2000],
                "evidence_score": result.get("evidence_score", 0),
                "categories": result.get("categories", []),
                "trace": result.get("trace", []),
                "sources": [{"score": round(s, 2), "comment": doc.get("comment", "")[:200]}
                           for s, doc in result.get("results", [])[:5]],
                "latency_ms": 0}
    except Exception as e:
        return {"error": str(e)}


def _run_bm25_cache(query: str, top_k: int = 10) -> Optional[dict]:
    """BM25 关键词检索 — dyx 贡献 (缓存复用)."""
    try:
        from _shared.cache import cache
        index = cache.get_bm25()
        df = cache.get_comments()
        raw = index.search(query, topk=top_k)
        items = []
        for doc_id, score in raw:
            row = df[df["_id"].astype(str) == doc_id]
            comment = str(row["comment"].values[0]) if not row.empty else ""
            items.append({"doc_id": doc_id, "score": round(score, 4), "comment": comment[:300]})
        return {"results": items, "method": "BM25+jieba", "total_docs": index.num_docs, "latency_ms": 0}
    except Exception as e:
        return {"error": str(e)}


# -- Main API (v2: 使用统一检索编排器) --

def route(query: str, top_k: int = 10,
          strategies: Optional[List[str]] = None,
          use_llm: bool = False) -> dict:
    """Layer 2 统一入口 (v2).

    使用 _shared/retriever 统一编排, 共享缓存避免重复加载.

    Args:
        query: natural language query
        top_k: results per strategy
        strategies: manual override (None = auto-detect via dyx intent classifier)
        use_llm: enable LLM-based intent classification

    Returns:
        {query, intent, router, strategies, results, meta}
    """
    start = time.time()
    routing = {"method": "keyword"}
    llm_attempted = False

    # 策略自动检测
    if strategies is None:
        if use_llm:
            llm_attempted = True
            llm_result = classify_intent_llm(query)
            if llm_result:
                intent = llm_result["intent"]
                routing = {"method": "llm", "model": llm_result["model"],
                          "reason": llm_result["reason"]}
            else:
                intent = classify_intent(query)
                routing = {"method": "keyword", "note": "no API key or network error"}
        else:
            # [dyx] 细粒度中文意图分类
            detailed = classify_intent_detailed(query)
            intent = detailed["primary"]
            strategies = detailed["strategies"]
            routing = {"method": "keyword", "intent_detail": detailed}
    else:
        intent_map = {"multimodal": "visual", "rag": "qa", "bm25": "keyword",
                     "fusion": "mixed", "all": "mixed"}
        intent = intent_map.get(strategies[0] if len(strategies) == 1 else "all", "mixed")

    # 通过统一检索器执行
    from _shared.retriever import retrieve as _retrieve
    retriever_result = _retrieve(query, top_k=top_k, strategies=strategies, use_llm=use_llm)

    results = retriever_result.get("results", {})

    try:
        from _shared.cache import cache
        stats = cache.get_retriever().stats
    except Exception:
        stats = {}

    output = {
        "query": query,
        "intent": intent,
        "router": routing,
        "strategies": strategies,
        "results": results,
        "meta": {
            "index_stats": stats,
            "total_latency_ms": round((time.time() - start) * 1000),
            "engines": retriever_result.get("engines", {}),
        },
    }
    if llm_attempted and routing.get("method") == "keyword":
        output["_hint"] = SETUP_HINT
    return output
# ===========================================================================
# [dyx 贡献] 细粒度中文意图分类 + 查询扩展
# ===========================================================================

_INTENT_CATEGORIES_ZH = {
    "facility":       ["设施", "设备", "装修", "房间", "游泳池", "健身房", "花园",
                        "停车场", "电梯", "空调", "热水", "无线", "wifi", "网络"],
    "price":          ["价格", "性价比", "费用", "划算", "值得", "贵", "便宜", "多少钱", "值", "优惠", "折扣", "房价"],
    "location":       ["位置", "交通", "周边", "地铁", "商圈", "距离", "出行",
                        "方便", "市中心", "景区"],
    "service":        ["服务", "前台", "态度", "专业", "热情", "效率", "办理",
                        "入住", "退房", "客服", "管家"],
    "food":           ["早餐", "餐饮", "餐厅", "美食", "食品", "自助餐", "品种", "菜品", "口味", "好吃"],
    "cleanliness":    ["卫生", "干净", "清洁", "整洁", "脏", "异味", "霉"],
    "quiet":          ["安静", "噪音", "隔音", "吵", "嘈杂", "睡眠"],
    "child_friendly": ["亲子", "儿童", "小孩", "孩子", "家庭", "乐园"],
    "elder_friendly": ["老人", "长辈", "轮椅", "无障碍", "适老"],
    "overall":        ["推荐", "体验", "感受", "总结", "评价", "怎么样", "如何"],
}

_INTENT_STRATEGY_MAP = {
    "facility":       "multimodal",
    "price":          "rag", "location":     "rag",
    "service":        "rag", "food":         "rag",
    "cleanliness":    "rag", "quiet":        "rag",
    "child_friendly": "rag", "elder_friendly": "rag",
    "overall":        "rag",
}


def classify_intent_detailed(query: str) -> dict:
    """细粒度中文意图分类 — 返回 {primary, categories, confidence, strategies}.

    基于 dyx 的关键词分类体系, 映射到现有策略.
    """
    q_lower = query.lower()
    scores = {}
    for cat, keywords in _INTENT_CATEGORIES_ZH.items():
        score = sum(1 for kw in keywords if kw in q_lower or kw.lower() in query)
        if score > 0:
            scores[cat] = score

    if not scores:
        return {"primary": "mixed", "categories": [], "confidence": 0.5,
                "strategies": ["multimodal", "rag"]}

    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_cats[0][0]
    confidence = sorted_cats[0][1] / max(sum(scores.values()), 1)

    strategy_set = set()
    for cat, _ in sorted_cats:
        s = _INTENT_STRATEGY_MAP.get(cat, "rag")
        strategy_set.add(s)
    strategies = list(strategy_set) if strategy_set else ["multimodal", "rag"]

    return {
        "primary": primary,
        "categories": [c for c, _ in sorted_cats],
        "confidence": round(min(confidence, 1.0), 4),
        "strategies": strategies,
    }


def expand_query(query: str, max_expansions: int = 2) -> list:
    """用 LLM 扩展查询, 生成 1~2 个相关子问题."""
    from _shared.llm import chat
    prompt = (
        "酒店评论搜索: " + query + "\n\n"
        "请生成 1~2 个相关的搜索意图, 帮助更全面地检索.\n"
        "每行一个, 尽量简洁."
    )
    try:
        result = chat([
            {"role": "system", "content": "你是酒店搜索助手, 只输出搜索词, 每行一个."},
            {"role": "user", "content": prompt},
        ], max_tokens=100, temperature=0.3)
        if result:
            expansions = [line.strip().strip("-").strip()
                         for line in result.strip().split("\n")
                         if line.strip()]
            seen = {query}
            queries = [query]
            for exp in expansions:
                if exp and exp not in seen:
                    queries.append(exp)
                    seen.add(exp)
                if len(queries) > max_expansions:
                    break
            return queries
    except Exception:
        pass
    return [query]
