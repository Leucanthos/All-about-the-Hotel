"""统一 LLM 接口 — 本地模型 > API > 回退.

优先级:
  1. 本地小模型 (Qwen2.5-0.5B-Instruct, ~1GB, 纯 CPU 可跑)
  2. DeepSeek API (需 DEEPSEEK_API_KEY)
  3. 无模型 (返回 None, 上游自行回退)

用法:
    from _shared.llm import chat, classify_json
    answer = chat([{"role": "user", "content": "..."}])
    intent = classify_json(system_prompt, user_query)
"""

import json
import os
import urllib.request
from typing import Optional

# -- local model (lazy load) --

_local_model = None
_local_tokenizer = None
_LOCAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _get_local():
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer
    # 快速检查模型是否已缓存 (纯本地，不联网)
    import os as _os
    cache_dir = _os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = _LOCAL_MODEL_ID.replace("/", "--")
    if not _os.path.exists(_os.path.join(cache_dir, "models--" + model_dir)):
        return None, None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[LLM] Loading local model (Qwen2.5-0.5B-Instruct)...")
        _local_tokenizer = AutoTokenizer.from_pretrained(_LOCAL_MODEL_ID, local_files_only=True)
        _local_model = AutoModelForCausalLM.from_pretrained(
            _LOCAL_MODEL_ID, torch_dtype="auto", device_map="auto", local_files_only=True
        )
        print("[LLM] Local model loaded.")
        return _local_model, _local_tokenizer
    except Exception:
        return None, None


def is_local_available() -> bool:
    """本地模型是否已加载."""
    m, _ = _get_local()
    return m is not None


def _get_api_key() -> Optional[str]:
    return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")


def _get_api_base() -> str:
    return (os.getenv("LLM_API_BASE") or "https://api.deepseek.com").rstrip("/")


def _get_api_model() -> str:
    return os.getenv("LLM_MODEL") or "deepseek-chat"


# -- core API --

def chat(messages: list[dict], max_tokens: int = 512,
         temperature: float = 0.2, prefer: str = "auto") -> Optional[str]:
    """通用对话接口. prefer: 'auto' | 'local' | 'api'.

    返回 None 表示无可用模型 (调用方自行回退).
    """
    # 1. 本地模型
    if prefer in ("auto", "local"):
        model, tok = _get_local()
        if model is not None:
            try:
                import torch
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tok(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_tokens,
                                            temperature=temperature, do_sample=temperature > 0,
                                            pad_token_id=tok.eos_token_id)
                return tok.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            except Exception:
                pass

    # 2. API
    if prefer in ("auto", "api"):
        api_key = _get_api_key()
        if api_key:
            try:
                payload = json.dumps({
                    "model": _get_api_model(), "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens,
                }, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    _get_api_base() + "/chat/completions", data=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"]
            except Exception:
                pass

    return None


def classify_json(system_prompt: str, user_query: str,
                  prefer: str = "auto") -> Optional[dict]:
    """分类任务 (小输出). 要求模型返回 JSON."""
    model, tok = _get_local()
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # 本地模型: 用 chat + 手动解析 JSON
    if prefer in ("auto", "local") and model is not None:
        raw = chat(msgs, max_tokens=128, temperature=0, prefer="local")
        if raw:
            try:
                # 提取 JSON
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
            except Exception:
                pass

    # API: 使用 response_format
    if prefer in ("auto", "api"):
        api_key = _get_api_key()
        if api_key:
            try:
                payload = json.dumps({
                    "model": _get_api_model(), "messages": msgs,
                    "temperature": 0, "max_tokens": 100,
                    "response_format": {"type": "json_object"},
                }, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    _get_api_base() + "/chat/completions", data=payload,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return json.loads(body["choices"][0]["message"]["content"])
            except Exception:
                pass

    return None
