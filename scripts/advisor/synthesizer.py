"""综合研判 + 结构化回复生成."""

from _shared.llm import chat
from .scenes import get_scene_label, DIMENSION_INFO


SYNTHESIS_PROMPT = """你是酒店入住决策顾问。用户场景：{scene_label}。

以下是各评估维度的住客反馈摘要：

{dimension_summaries}

请综合判断，按以下 JSON 格式返回：
{{
    "judgments": [
        {{"dimension": "网络", "verdict": "好|风险|注意", "reason": "一句话理由"}}
    ],
    "overall": "推荐|谨慎推荐|不推荐",
    "conclusion": "一句话总体结论",
    "suggestions": ["建议1", "建议2", "建议3"]
}}
只返回 JSON，不要多余文字。"""


def synthesize(scene: str, dimension_results: list) -> dict:
    """综合研判各维度结果，生成结构化建议.

    Args:
        scene: 场景 key
        dimension_results: [{dim_key, query, raw_result}]

    Returns:
        {judgments, overall, conclusion, suggestions}
    """
    # 构建摘要文本
    summaries = []
    for dr in dimension_results:
        dim_key = dr["dim_key"]
        info = DIMENSION_INFO.get(dim_key, {"label": dim_key, "emoji": "❓"})
        label = info["label"]
        raw = dr.get("raw_result", {})

        # 从检索结果中提取关键信息
        evidence = _extract_evidence(raw)
        summaries.append(f"{label}（查询：{dr['query']}）\n{evidence}")

    dim_text = "\n\n".join(summaries)
    scene_label = get_scene_label(scene)

    prompt = SYNTHESIS_PROMPT.format(
        scene_label=scene_label,
        dimension_summaries=dim_text,
    )

    try:
        result = _llm_judge(prompt)
        if result:
            return result
    except Exception:
        pass

    return _fallback_judgment(dimension_results)


def _extract_evidence(raw: dict) -> str:
    """从检索结果中提取关键证据文本."""
    parts = []

    # 多模态结果
    results = raw.get("results", {})
    for strategy_name, strategy_result in results.items():
        if isinstance(strategy_result, dict):
            # RAG / BM25 结果
            sources = strategy_result.get("sources", []) or strategy_result.get("results", [])
            if sources and isinstance(sources, list):
                for s in sources[:3]:
                    comment = s.get("comment", "") or s.get("content", "")
                    if comment:
                        parts.append(f"[{strategy_name}]「{comment[:150]}」")

            # 图片结果
            images = strategy_result.get("images", [])
            if images:
                parts.append(f"[多模态] 找到 {len(images)} 张相关图片")

            answer = strategy_result.get("answer", "")
            if answer:
                parts.append(f"[分析] {answer[:200]}")

    if not parts:
        parts.append("（该维度暂无详细证据）")

    return "\n".join(parts[:5])


def _llm_judge(prompt: str) -> dict:
    """调用 LLM 进行综合研判."""
    from _shared.llm import chat
    result = chat([
        {"role": "system", "content": "你是酒店入住决策顾问，只输出 JSON。"},
        {"role": "user", "content": prompt},
    ], max_tokens=800, temperature=0.1, prefer="api")

    if result:
        import json
        # 提取 JSON
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(result[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _fallback_judgment(dimension_results: list) -> dict:
    """LLM 不可用时的模板回退."""
    judgments = []
    for dr in dimension_results:
        dim_key = dr["dim_key"]
        info = DIMENSION_INFO.get(dim_key, {"label": dim_key, "emoji": "❓"})
        judgments.append({
            "dimension": info["label"],
            "verdict": "注意",
            "reason": "请结合实际入住体验判断",
        })
    return {
        "judgments": judgments,
        "overall": "谨慎推荐",
        "conclusion": "建议结合具体需求进一步确认",
        "suggestions": ["查看具体评论", "联系酒店确认细节"],
    }


def format_reply(scene: str, scene_label: str, synthesis: dict, dimension_results: list) -> str:
    """生成结构化 Markdown 回复."""
    lines = [f"好的，**{scene_label}场景**。帮你逐项评估：", ""]

    # 各维度判断
    for dr in dimension_results:
        dim_key = dr["dim_key"]
        info = DIMENSION_INFO.get(dim_key, {"label": dim_key, "emoji": "❓"})
        label = info["label"]
        emoji = info["emoji"]

        # 找对应的 judgment
        verdict = "注意"
        reason = ""
        for j in synthesis.get("judgments", []):
            if j.get("dimension", "").strip() == label.strip():
                verdict = j.get("verdict", "注意")
                reason = j.get("reason", "")
                break

        lines.append(f"{emoji} **{label}** — {verdict}")
        if reason:
            lines.append(f"> {reason}")

        # 引用住客原话
        raw = dr.get("raw_result", {})
        quote = _extract_one_quote(raw)
        if quote:
            lines.append(f"> 「{quote}」")

        lines.append("")

    # 结论
    lines.append("---")
    lines.append(f"📋 **结论：{synthesis.get('conclusion', '请综合判断')}**")
    for s in synthesis.get("suggestions", []):
        lines.append(f"- {s}")

    return "\n".join(lines)


def _extract_one_quote(raw: dict) -> str:
    """从检索结果中提取一条住客原话."""
    results = raw.get("results", {})
    for strategy_name, strategy_result in results.items():
        if isinstance(strategy_result, dict):
            sources = strategy_result.get("sources", []) or strategy_result.get("results", [])
            if sources and isinstance(sources, list):
                for s in sources:
                    comment = s.get("comment", "") or s.get("content", "")
                    if comment and len(comment) > 10:
                        return comment[:120]
    return ""
