"""Agentic RAG 公开 API — 一站式问答入口 (SUN Yuhao)."""
import json
from pathlib import Path

from _shared.text import normalize
from _shared.data import COMMENTS_PATH, SUMMARIES_PATH
from .engine import AgenticRAG, HotelCorpus, build_context

NEGATIVE_WORDS = [
    "老旧", "噪音", "排队", "卫生", "不便", "异味", "施工",
    "慢", "无奈", "坏", "吐槽", "差", "投诉", "不专业", "混乱", "失望",
]


def fallback_answer(rag_result):
    """本地模板回答 — 无需 LLM."""
    context = build_context(rag_result)
    cats = "、".join(rag_result["categories"][:4]) or "综合体验"
    risks, positives = [], []

    for _, doc in rag_result["results"][:5]:
        text = doc["comment"]
        try:
            score_value = float(doc.get("score") or 0)
        except ValueError:
            score_value = 0
        if any(k in text for k in NEGATIVE_WORDS) or score_value < 4.0:
            risks.append(text[:90])
        elif score_value >= 4.5:
            positives.append(text[:90])

    if not positives:
        positives = [
            normalize(item.get("summary", ""))[:90]
            for item in rag_result["summaries"][:2]
            if item.get("summary")
        ]
    positives = positives[:2] or [doc["comment"][:90] for _, doc in rag_result["results"][:2]]
    risks = risks[:2] or ["当前命中评论中负面证据较少，建议结合具体房型和入住日期确认。"]

    return (
        f"结论：这个问题主要涉及{cats}。证据覆盖分为 {rag_result['evidence_score']}，可以作为回答依据。\n\n"
        f"主要亮点：\n- " + "\n- ".join(positives) + "\n\n"
        f"需要提示的风险：\n- " + "\n- ".join(risks) + "\n\n"
        f"建议：预订时备注具体需求，例如安静房、翻新房型、老人小孩出行便利；入住高峰期提前到店或预留办理时间。\n\n"
        f"参考证据节选：\n{context[:900]}"
    )


def answer_with_optional_llm(rag_result, use_llm=False):
    """LLM 生成回答 — 本地模型 > DeepSeek API > 模板回退."""
    if not use_llm:
        return fallback_answer(rag_result)

    from _shared.llm import chat
    messages = [
        {"role": "system", "content": "你是严谨的酒店评论问答助手，只能依据证据回答。"},
        {"role": "user", "content": (
            f"用户问题：{rag_result['query']}\n\n"
            f"Agent 检索轨迹：{json.dumps(rag_result['trace'], ensure_ascii=False)}\n\n"
            f"证据：\n{build_context(rag_result)}\n\n"
            "请输出：结论、证据依据、风险提醒、行动建议。"
        )},
    ]
    try:
        result = chat(messages, max_tokens=512)
        if result:
            return result
    except Exception:
        pass
    return fallback_answer(rag_result)


def answer_question(
    query: str,
    top_k: int = 6,
    use_llm: bool = False,
    comments_path: Path = COMMENTS_PATH,
    summaries_path: Path = SUMMARIES_PATH,
    max_docs: int = 4000,
):
    """运行 Agentic RAG 流水线，返回结构化回答."""
    corpus = HotelCorpus(
        comments_path=comments_path, summaries_path=summaries_path, max_docs=max_docs
    )
    agent = AgenticRAG(corpus)
    result = agent.run(query, top_k=top_k)
    answer = answer_with_optional_llm(result, use_llm)
    return {"answer": answer, "result": result, "docs": len(corpus.docs)}
