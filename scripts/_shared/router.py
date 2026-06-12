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


# -- Engines --

def _run_multimodal(query: str, top_k: int = 10) -> Optional[dict]:
    try:
        from leuca.multimodal.api import MultimodalAPI
        api = MultimodalAPI()
        r = api.prompt2image(query, topK=top_k)
        return {"images": r["images"], "scores": [round(s, 4) for s in r["score"]],
                "method": "CLIP", "latency_ms": 0}
    except Exception as e:
        return {"error": str(e)}


def _run_rag(query: str, top_k: int = 8) -> Optional[dict]:
    try:
        from yuhao.agentic.api import answer_question
        has_key = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY"))
        output = answer_question(query, top_k=top_k, use_llm=has_key)
        result = output["result"]
        return {"answer": output["answer"][:2000],
                "evidence_score": result["evidence_score"],
                "categories": result["categories"],
                "sources": [{"score": round(s, 2), "comment": doc.get("comment", "")[:200]}
                           for s, doc in result.get("results", [])[:5]],
                "latency_ms": 0}
    except Exception as e:
        return {"error": str(e)}


# -- Main API --

def route(query: str, top_k: int = 10,
          strategies: Optional[List[str]] = None,
          use_llm: bool = False) -> dict:
    """Layer 2 entry point.

    Args:
        query: natural language query
        top_k: results per strategy
        strategies: manual override (None = auto-detect)
        use_llm: enable LLM-based intent classification (needs API key)

    Returns:
        {query, intent, router, strategies, results, meta}
    """
    start = time.time()
    routing = {"method": "keyword"}
    llm_attempted = False

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
            intent = classify_intent(query)
        strategies = _intent_to_strategies(intent)
    else:
        intent_map = {"multimodal": "visual", "rag": "qa", "all": "mixed"}
        intent = intent_map.get(strategies[0] if len(strategies) == 1 else "all", "mixed")

    results = {}
    for name in strategies:
        t0 = time.time()
        r = _run_multimodal(query, top_k) if name == "multimodal" \
            else _run_rag(query, min(top_k, 8))
        if r:
            r["latency_ms"] = round((time.time() - t0) * 1000)
        results[name] = r

    try:
        from leuca.multimodal.api import MultimodalAPI
        stats = MultimodalAPI().stats
    except Exception:
        stats = {}

    output = {
        "query": query, "intent": intent, "router": routing,
        "strategies": strategies, "results": results,
        "meta": {"index_stats": stats, "total_latency_ms": round((time.time() - start) * 1000)},
    }
    if llm_attempted and routing["method"] == "keyword":
        output["_hint"] = SETUP_HINT
    return output
