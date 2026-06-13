"""场景顾问 Agent — 主控接口.

使用流程:
    from advisor.agent import advise
    result = advise("下周出差三天要开视频会议，合适吗？")
    for step in result["steps"]:
        print(step["msg"])
    print(result["reply"])
"""

import time
from typing import Optional

from _shared.retriever import retrieve
from .recognizer import recognize
from .scenes import get_dimensions_for_scene, get_scene_label, QUERY_TEMPLATES
from .synthesizer import synthesize, format_reply


def advise(
    user_input: str,
    session_id: str = "default",
    verbose: bool = True,
) -> dict:
    """一站式场景顾问.

    Args:
        user_input: 用户自然语言
        session_id: 会话 ID（预留多轮记忆）
        verbose: 是否返回 step 明细

    Returns:
        {scene, confidence, dimensions, steps, reply, metrics}
    """
    start = time.time()
    steps = []
    def _log(phase, msg, detail=None):
        if verbose:
            steps.append({"phase": phase, "msg": msg, "detail": detail})

    # ═══════════════════════════════════════════════════
    # ① 场景识别
    # ═══════════════════════════════════════════════════
    _log("scene", "🔍 正在分析你的出行场景...")
    scene_info = recognize(user_input)
    scene = scene_info["scene"]
    scene_label = get_scene_label(scene)
    confidence = scene_info["confidence"]

    _log("scene_done",
         f"📋 识别为「{scene_label}」场景 (置信度 {confidence:.0%})",
         scene_info)

    if scene == "general":
        # 场景无法识别，走通用检索
        _log("search", "📎 场景不明确，改用通用检索...")
        raw = retrieve(user_input, top_k=5, use_llm=False)
        _log("done", "✅ 检索完成")
        return {
            "scene": "general",
            "confidence": 0,
            "dimensions": [],
            "steps": steps,
            "reply": _format_general(raw),
            "raw": raw,
            "metrics": {"total_latency": round(time.time() - start, 2)},
        }

    is_practical = (scene == "practical")

    if is_practical:
        # 事实查证型：提取关键词 → BM25 检索 → 去重展示
        subtype_key = scene_info.get("specifics", {}).get("subtype", "general")
        search_query = _make_search_query(user_input)
        _log("search", f"🔎 正在检索「{search_query}」...")
        raw = retrieve(search_query, top_k=8, strategies=["rag", "bm25"], use_llm=False)
        _log("synthesize", "📋 正在汇总...")
        reply = _format_lookup(user_input, raw, subtype_key)
        _log("done", "✅ 查询完成")
        return {
            "scene": scene, "scene_label": scene_label, "confidence": confidence,
            "dimensions": [], "steps": steps,
            "judgments": [], "overall": "", "conclusion": "", "suggestions": [],
            "reply": reply,
            "metrics": {"total_latency": round(time.time() - start, 2)},
        }

    # ═══════════════════════════════════════════════════
    # ② 维度展开 (advisory 场景)
    # ═══════════════════════════════════════════════════
    dimensions = get_dimensions_for_scene(scene)
    core = [d for d in dimensions if d["level"] == "●"]
    secondary = [d for d in dimensions if d["level"] == "○"]

    _log("dimensions",
         f"📐 拆解为 {len(core)} 个核心维度 + {len(secondary)} 个次要维度")

    if scene_info.get("specifics"):
        selected = core.copy()
        specifics_str = str(scene_info["specifics"]).lower()
        for d in secondary:
            if any(kw in specifics_str for kw in [d["key"], d["label"]]):
                selected.append(d)
    else:
        selected = core

    for d in selected:
        level_tag = "●核心" if d["level"] == "●" else "○次要"
        _log("dim_item", f"  {d['emoji']} {d['label']} [{level_tag}]")

    # ═══════════════════════════════════════════════════
    # ③ 生成维度 query
    # ═══════════════════════════════════════════════════
    dimension_queries = []
    for d in selected:
        # 使用场景特定的 query 模板 + specifics 填充
        template = d["query_template"]
        specifics = scene_info.get("specifics", {})
        query = _fill_template(template, specifics, d["label"], user_input)
        dimension_queries.append({
            "dim_key": d["key"],
            "label": d["label"],
            "emoji": d["emoji"],
            "query": query,
        })

    for dq in dimension_queries:
        _log("query", f"  {dq['emoji']} {dq['label']}: 「{dq['query']}」")

    # ═══════════════════════════════════════════════════
    # ④ 并行检索
    # ═══════════════════════════════════════════════════
    total_dims = len(dimension_queries)
    _log("search", f"🔎 并发检索 {total_dims} 个维度...")

    dimension_results = []
    for i, dq in enumerate(dimension_queries):
        _log("searching", f"  [{i+1}/{total_dims}] 查询 {dq['label']}...")
        raw = retrieve(dq["query"], top_k=5, strategies=None, use_llm=False)
        dimension_results.append({
            "dim_key": dq["dim_key"],
            "label": dq["label"],
            "emoji": dq["emoji"],
            "query": dq["query"],
            "raw_result": raw,
        })

    _log("search_done", f"✅ {total_dims} 个维度检索完成")

    # ═══════════════════════════════════════════════════
    # ⑤ 综合研判
    # ═══════════════════════════════════════════════════
    _log("synthesize", "🧠 正在综合研判各维度结果...")
    synthesis = synthesize(scene, dimension_results)
    _log("synthesize_done",
         f"📊 综合结论: {synthesis.get('overall', '请判断')}",
         synthesis)

    # ═══════════════════════════════════════════════════
    # ⑥ 生成回复
    # ═══════════════════════════════════════════════════
    reply = format_reply(scene, scene_label, synthesis, dimension_results)
    _log("done", "✅ 评估完成")

    return {
        "scene": scene,
        "scene_label": scene_label,
        "confidence": confidence,
        "dimensions": [dq["dim_key"] for dq in dimension_queries],
        "steps": steps,
        "judgments": synthesis.get("judgments", []),
        "overall": synthesis.get("overall", ""),
        "conclusion": synthesis.get("conclusion", ""),
        "suggestions": synthesis.get("suggestions", []),
        "reply": reply,
        "metrics": {"total_latency": round(time.time() - start, 2)},
    }


def advise_stream(user_input: str, session_id: str = "default"):
    """流式场景顾问 — 逐步 yield 每个流程节点 (供 SSE 端点使用).

    用法:
        for event in advise_stream("下周出差三天要开视频会议，合适吗？"):
            # event: {"phase": "...", "msg": "..."}
            # phase="done" 时附带 reply/judgments/overall 等完整结果
            print(event["msg"])
    """
    start = time.time()

    # ═══════════════════════════════════════════════════
    # ① 场景识别
    # ═══════════════════════════════════════════════════
    yield {"phase": "recognizing", "msg": "正在分析你的出行场景..."}
    scene_info = recognize(user_input)
    scene = scene_info["scene"]
    scene_label = get_scene_label(scene)
    confidence = scene_info["confidence"]
    router_method = scene_info.get("router", "keyword")

    yield {
        "phase": "scene_done",
        "msg": f"识别为「{scene_label}」场景 (置信度 {confidence:.0%}, 路由: {router_method})",
        "data": scene_info,
    }

    if scene == "general":
        yield {"phase": "search", "msg": "场景不明确，改用通用检索..."}
        raw = retrieve(user_input, top_k=5, use_llm=False)
        yield {
            "phase": "done",
            "msg": "评估完成",
            "reply": _format_general(raw),
            "metrics": {"total_latency": round(time.time() - start, 2)},
        }
        return

    # ── practical 场景: 直接检索，不走维度展开 ──
    is_practical = (scene == "practical")

    if is_practical:
        # 事实查证型：提取关键词 → BM25 检索 → 去重展示
        subtitle = scene_info.get("specifics", {}).get("subtype", "general")
        search_query = _make_search_query(user_input)
        yield {"phase": "search",
               "msg": f"正在检索「{search_query}」..."}
        raw = retrieve(search_query, top_k=8, strategies=["rag", "bm25"], use_llm=False)
        yield {"phase": "synthesizing", "msg": "正在汇总相关住客反馈..."}
        reply = _format_lookup(user_input, raw, subtitle)
        yield {"phase": "synthesize_done", "msg": "汇总完成"}
        yield {
            "phase": "done",
            "msg": "查询完成",
            "scene": scene,
            "scene_label": scene_label,
            "confidence": confidence,
            "reply": reply,
            "judgments": [],
            "overall": "",
            "conclusion": "",
            "suggestions": [],
            "metrics": {"total_latency": round(time.time() - start, 2)},
        }
        return

    # ═══════════════════════════════════════════════════
    # ② 维度拆解 + query 改写 (advisory 场景)
    # ═══════════════════════════════════════════════════
    dimensions = get_dimensions_for_scene(scene)
    core = [d for d in dimensions if d["level"] == "●"]
    secondary = [d for d in dimensions if d["level"] == "○"]

    yield {"phase": "dimensions",
           "msg": f"拆解为 {len(core)} 个核心维度 + {len(secondary)} 个次要维度"}

    if scene_info.get("specifics"):
        selected = core.copy()
        specifics_str = str(scene_info["specifics"]).lower()
        for d in secondary:
            if any(kw in specifics_str for kw in [d["key"], d["label"]]):
                selected.append(d)
    else:
        selected = core

    # 生成维度 query（改写）
    dimension_queries = []
    for d in selected:
        template = d["query_template"]
        specifics = scene_info.get("specifics", {})
        query = _fill_template(template, specifics, d["label"], user_input)
        level_tag = "核心" if d["level"] == "●" else "次要"
        dimension_queries.append({
            "dim_key": d["key"],
            "label": d["label"],
            "query": query,
            "level_tag": level_tag,
        })
        yield {"phase": "query_rewrite",
               "msg": f"  {d['label']} [{level_tag}] → 「{query}」"}

    # ═══════════════════════════════════════════════════
    # ③ 检索 (含分流信息)
    # ═══════════════════════════════════════════════════
    total_dims = len(dimension_queries)
    yield {"phase": "search", "msg": f"并发检索 {total_dims} 个维度..."}

    dimension_results = []
    for i, dq in enumerate(dimension_queries):
        yield {"phase": "searching",
               "msg": f"  [{i+1}/{total_dims}] 查询 {dq['label']} -> 召回中..."}
        raw = retrieve(dq["query"], top_k=5, strategies=None, use_llm=False)
        strategies_used = raw.get("strategies", [])
        strategies_str = ", ".join(strategies_used) if strategies_used else "auto"
        dimension_results.append({
            "dim_key": dq["dim_key"],
            "label": dq["label"],
            "query": dq["query"],
            "raw_result": raw,
        })
        yield {"phase": "search_done",
               "msg": f"  [{i+1}/{total_dims}] {dq['label']} -> 召回方式: {strategies_str}"}

    # ═══════════════════════════════════════════════════
    # ④ 综合研判
    # ═══════════════════════════════════════════════════
    yield {"phase": "synthesizing", "msg": "正在综合研判各维度结果..."}
    synthesis = synthesize(scene, dimension_results)

    yield {"phase": "synthesize_done",
           "msg": f"综合结论: {synthesis.get('overall', '请判断')}",
           "data": synthesis}

    # ═══════════════════════════════════════════════════
    # ⑤ 生成最终回复
    # ═══════════════════════════════════════════════════
    reply = format_reply(scene, scene_label, synthesis, dimension_results)

    yield {
        "phase": "done",
        "msg": "评估完成",
        "scene": scene,
        "scene_label": scene_label,
        "confidence": confidence,
        "reply": reply,
        "judgments": synthesis.get("judgments", []),
        "overall": synthesis.get("overall", ""),
        "conclusion": synthesis.get("conclusion", ""),
        "suggestions": synthesis.get("suggestions", []),
        "metrics": {"total_latency": round(time.time() - start, 2)},
    }


def _fill_template(template: str, specifics: dict, label: str, user_input: str) -> str:
    """用 specifics 填充 query 模板."""
    if template:
        return template
    # 无模板时的 fallback
    return f"{label}怎么样？{user_input[:50]}"


def _format_general(raw: dict) -> str:
    """场景无法识别时的通用回复."""
    lines = ["未能识别你的出行场景，以下是通用检索结果：", ""]
    results = raw.get("results", {})
    for name, result in results.items():
        if isinstance(result, dict):
            items = result.get("results", []) or result.get("sources", [])
            if items:
                lines.append(f"**{name}** 相关评论：")
                for item in items[:3]:
                    comment = item.get("comment", "") or item.get("content", "")
                    if comment:
                        lines.append(f"- {comment[:120]}")
                lines.append("")
    if len(lines) <= 2:
        lines.append("（暂无结果，试试换个问法）")
    return "\n".join(lines)


def _format_lookup(user_input: str, raw: dict, subtype: str = "") -> str:
    """事实查证回复 — 直接检索，去重 + 相关性过滤."""
    seen = set()
    items = []

    results = raw.get("results", {})
    for strategy_name in ["rag", "bm25"]:
        strategy_result = results.get(strategy_name, {})
        if not isinstance(strategy_result, dict):
            continue
        sources = (strategy_result.get("sources", []) or
                   strategy_result.get("results", []))
        for s in sources:
            comment = (s.get("comment", "") or s.get("content", "")).strip()
            score = s.get("score", 0)
            if not comment or len(comment) < 15:
                continue
            # 去重：基于评论前 60 字
            key = comment[:60]
            if key in seen:
                continue
            seen.add(key)
            # 简单相关性：至少有一个核心词命中
            items.append((score, comment))

    # 按分排序，取 top 5
    items.sort(key=lambda x: x[0], reverse=True)
    top_items = items[:5]

    # 提取问题中的核心词（去掉疑问/语气词后）
    import re as _re
    core_terms = [t for t in _re.split(r"[，。？?？\s]+", _make_search_query(user_input)) if len(t) >= 2]

    # 过滤：评论中至少包含一个核心词
    relevant = []
    for score, comment in top_items:
        if any(term in comment for term in core_terms):
            relevant.append((score, comment))
        if len(relevant) >= 3:
            break

    lines = [f"关于「{user_input}」的住客反馈：", ""]
    if not relevant:
        lines.append("评论中暂未找到直接相关信息。")
        lines.append("建议直接联系酒店前台确认。")
        return "\n".join(lines)

    for score, comment in relevant:
        lines.append(f"> 「{comment[:200]}」")
    lines.append("")
    lines.append("---")
    lines.append("以上来自住客真实评论，具体政策建议直接联系酒店前台确认。")
    return "\n".join(lines)


def _subtype_label(subtype: str) -> str:
    """subtype → 中文标签."""
    labels = {
        "checkin": "入住时间", "checkout": "退房效率", "luggage": "行李寄存",
        "shuttle": "接送机", "payment": "支付方式", "room_amenity": "客房设施",
        "dietary": "饮食限制", "accessibility": "无障碍", "health_safety": "健康安全",
        "tech_service": "技术服务", "service_misc": "生活服务", "pet": "宠物政策",
        "special_req": "特殊需求",
    }
    return labels.get(subtype, "综合查询")


def _make_search_query(user_input: str) -> str:
    """把自然语言问题转成 BM25 友好的关键词查询.

    「双人房可以住三人吗」→ 「双人房 三人 入住」
    「半夜12:30到达可以入住吗」→ 「半夜 到达 入住」
    """
    # 去掉疑问词和语气词，保留实词
    noise = {"可以", "吗", "的", "了", "是", "有", "能", "会", "要",
             "什么", "怎么", "怎么样", "多少", "几", "不", "没", "这",
             "那", "我", "你", "他", "她", "在", "和", "与", "或",
             "一个", "一下", "给我", "帮我", "请问", "酒店", "提供",
             "入住", "房间", "这家", "你们", "他们", "可不可以",
             "能不能", "会不会", "需不需要"}
    words = user_input
    for w in sorted(noise, key=len, reverse=True):
        words = words.replace(w, " ")
    # 去空白，去重复
    parts = [p.strip() for p in words.split() if len(p.strip()) >= 1]
    seen = set()
    result = []
    for p in parts:
        if p not in seen:
            result.append(p)
            seen.add(p)
    return " ".join(result) if result else user_input


