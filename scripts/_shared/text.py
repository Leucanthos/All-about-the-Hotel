"""共享文本处理 — leuca / yuhao 统一使用."""
import ast
import csv
import re
from pathlib import Path


def normalize(text: str) -> str:
    """压缩空白字符."""
    return " ".join((text or "").split())


def short_text(text: str, max_chars: int = 420) -> str:
    """截短文本."""
    return normalize(text)[:max_chars]


def parse_categories(value: str) -> list[str]:
    """解析 '['a','b']' 格式的类别字符串."""
    try:
        return [str(x) for x in ast.literal_eval(value or "[]")]
    except Exception:
        return []


def char_terms(text: str) -> list[str]:
    """中文分词粒度: 英文词 + 单字 + 二元组."""
    text = normalize(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    chinese = re.findall(r"[一-鿿]", text)
    bigrams = ["".join(chinese[i: i + 2]) for i in range(max(0, len(chinese) - 1))]
    return words + chinese + bigrams


def read_csv(path: Path):
    """读取 CSV (utf-8-sig)."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)
