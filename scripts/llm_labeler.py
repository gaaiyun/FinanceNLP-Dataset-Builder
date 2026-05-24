"""LLM 驱动的金融文本自定义标注器。

v1 的 ``data_processor.py`` 内置了固定的 FinBERT-based 情感标注 + 规则提取的
公司名 / 人名 / 金额，但只能产出固定的标签 schema。v2 允许用户用一句话描述
标签方案（"标注情绪：positive/negative/neutral；标注事件类型：earnings/M&A/
regulatory；标注影响公司"），LLM 按 schema 一次产出。

设计取舍：
- 标签 schema 用 JSON，LLM 也按 JSON 输出，便于程序化处理
- 缺 API key 时直接 raise（不静默返回空标签）—— 标注是 v2 的核心，
  没 LLM 就该走 v1 的规则标注
- 提供 ``batch_label`` 串行调 LLM，按 schema 校验输出
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


LLMBackend = Literal["openai", "anthropic", "deepseek"]


class LLMNotAvailable(RuntimeError):
    pass


@dataclass
class LabelScheme:
    """标签方案：字段名 → 字段类型 / 取值约束。"""
    fields: Dict[str, Any]
    """例如：
    {
        "sentiment": ["positive", "negative", "neutral"],
        "event_type": ["earnings", "M&A", "regulatory", "product", "other"],
        "affected_companies": "list[str]",
        "confidence": "float[0-1]",
    }
    """
    description: str = ""

    def to_prompt_block(self) -> str:
        """渲染成 LLM 看得懂的字段说明。"""
        lines = []
        for name, spec in self.fields.items():
            if isinstance(spec, list):
                opts = "/".join(str(s) for s in spec)
                lines.append(f"  - {name}: 单选 ({opts})")
            elif isinstance(spec, str):
                lines.append(f"  - {name}: {spec}")
            else:
                lines.append(f"  - {name}: {spec}")
        result = "标签字段：\n" + "\n".join(lines)
        if self.description:
            result = self.description + "\n\n" + result
        return result


@dataclass
class LabeledSample:
    text: str
    labels: Dict[str, Any]
    backend: str
    raw_response: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "labels": self.labels,
            "backend": self.backend,
        }


class LLMClient:
    def __init__(self, backend: LLMBackend = "deepseek",
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: float = 60.0):
        self.backend = backend
        self.timeout = timeout
        self.api_key = api_key or self._default_key(backend)
        self.base_url = base_url or self._default_base_url(backend)
        self.model = model or self._default_model(backend)

    @staticmethod
    def _default_key(backend: LLMBackend) -> Optional[str]:
        return {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        }.get(backend)

    @staticmethod
    def _default_base_url(backend: LLMBackend) -> Optional[str]:
        return {"deepseek": "https://api.deepseek.com/v1"}.get(backend)

    @staticmethod
    def _default_model(backend: LLMBackend) -> str:
        return {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "deepseek": "deepseek-chat",
        }.get(backend, "gpt-4o-mini")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        if not self.is_available():
            raise LLMNotAvailable(
                f"{self.backend} backend 缺 API key（环境变量 "
                f"{self.backend.upper()}_API_KEY）"
            )
        if self.backend == "anthropic":
            return self._call_anthropic(system, user, temperature)
        return self._call_openai_compat(system, user, temperature)

    def _call_openai_compat(self, system: str, user: str, temperature: float) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMNotAvailable("缺 openai SDK：pip install openai") from e
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, system: str, user: str, temperature: float) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise LLMNotAvailable("缺 anthropic SDK：pip install anthropic") from e
        client = Anthropic(api_key=self.api_key, timeout=self.timeout)
        resp = client.messages.create(
            model=self.model, max_tokens=2048, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text if resp.content else ""


# --- 输出解析 ----------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _parse_labels(raw: str, scheme: LabelScheme) -> Dict[str, Any]:
    """从 LLM 输出抽 JSON 并按 schema 校验。

    校验是 best-effort：单选字段不在选项里 → 设 None；list 字段不是 list → 设 []。
    """
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 输出不是合规 JSON: {e}\n原始：{raw[:200]}") from e
    if not isinstance(data, dict):
        raise ValueError(f"期望 JSON object，得到 {type(data).__name__}")

    out: Dict[str, Any] = {}
    for name, spec in scheme.fields.items():
        val = data.get(name)
        if isinstance(spec, list):
            # 单选字段
            if val in spec:
                out[name] = val
            else:
                out[name] = None
        elif isinstance(spec, str) and spec.lower().startswith("list"):
            out[name] = val if isinstance(val, list) else []
        elif isinstance(spec, str) and "float" in spec.lower():
            try:
                out[name] = float(val) if val is not None else None
            except (TypeError, ValueError):
                out[name] = None
        elif isinstance(spec, str) and "int" in spec.lower():
            try:
                out[name] = int(val) if val is not None else None
            except (TypeError, ValueError):
                out[name] = None
        else:
            out[name] = val
    return out


# --- LLMLabeler --------------------------------------------------------------

class LLMLabeler:
    """主入口：把一段文本按 LabelScheme 转成结构化标签。"""

    def __init__(self, llm_client: Optional[LLMClient] = None,
                 backend: LLMBackend = "deepseek"):
        self.llm_client = llm_client or LLMClient(backend=backend)

    def label_one(self, text: str, scheme: LabelScheme) -> LabeledSample:
        if not self.llm_client.is_available():
            raise LLMNotAvailable(
                "LLM 没配 key。v2 的 LLMLabeler 强依赖 LLM；"
                "v1 的 data_processor.DataProcessor 提供规则标注作为 fallback。"
            )
        system = (
            "你是一名金融 NLP 数据标注员。按指定的标签字段方案给文本打标签。"
            "只输出 JSON object，字段严格按要求，不要任何前后缀说明。"
        )
        user = (
            f"{scheme.to_prompt_block()}\n\n"
            f"待标注文本：\n{text}\n\n"
            f"按上述字段方案输出 JSON。"
        )
        raw = self.llm_client.chat(system, user, temperature=0.1)
        labels = _parse_labels(raw, scheme)
        return LabeledSample(
            text=text, labels=labels,
            backend=f"llm:{self.llm_client.backend}",
            raw_response=raw,
        )

    def label_batch(self, texts: List[str], scheme: LabelScheme,
                    skip_errors: bool = True) -> List[LabeledSample]:
        """串行 label。批量大时建议外部加并发。"""
        results = []
        for t in texts:
            try:
                results.append(self.label_one(t, scheme))
            except (ValueError, LLMNotAvailable) as e:
                if not skip_errors:
                    raise
                # 错误样本占位（labels=None 字典）
                results.append(LabeledSample(
                    text=t, labels={k: None for k in scheme.fields},
                    backend="error", raw_response=str(e),
                ))
        return results


# --- 常用预设 schema --------------------------------------------------------

SENTIMENT_3WAY = LabelScheme(
    fields={"sentiment": ["positive", "negative", "neutral"]},
    description="金融文本情绪分类（仅情绪一项）。",
)

EVENT_TYPE = LabelScheme(
    fields={
        "sentiment": ["positive", "negative", "neutral"],
        "event_type": ["earnings", "M&A", "regulatory", "product", "macro", "other"],
        "affected_companies": "list[str]",
    },
    description="金融文本：情绪 + 事件类型 + 涉及公司。",
)


PRESETS = {
    "sentiment_3way": SENTIMENT_3WAY,
    "event_type": EVENT_TYPE,
}
