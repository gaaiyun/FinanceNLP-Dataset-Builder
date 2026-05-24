"""llm_labeler.py 测试 —— mock LLM。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.llm_labeler import (
    PRESETS,
    LabelScheme,
    LabeledSample,
    LLMClient,
    LLMLabeler,
    LLMNotAvailable,
    SENTIMENT_3WAY,
    _parse_labels,
    _strip_code_fences,
)


# --- LLMClient ----------------------------------------------------------------

def test_default_models():
    assert LLMClient(backend="openai", api_key="x").model == "gpt-4o-mini"
    assert LLMClient(backend="anthropic", api_key="x").model == "claude-3-5-haiku-20241022"
    assert LLMClient(backend="deepseek", api_key="x").model == "deepseek-chat"


def test_deepseek_base_url():
    c = LLMClient(backend="deepseek", api_key="x")
    assert c.base_url == "https://api.deepseek.com/v1"


def test_is_available_with_key():
    assert LLMClient(backend="deepseek", api_key="sk-test").is_available()


def test_chat_raises_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = LLMClient(backend="deepseek")
    with pytest.raises(LLMNotAvailable):
        c.chat("sys", "user")


def test_picks_up_env_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    c = LLMClient(backend="deepseek")
    assert c.api_key == "sk-from-env"


# --- _strip_code_fences -------------------------------------------------------

def test_strip_code_fences_json_block():
    assert _strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_bare_block():
    assert _strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fences_no_fence():
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


# --- _parse_labels ------------------------------------------------------------

def test_parse_labels_valid():
    raw = '{"sentiment": "positive"}'
    out = _parse_labels(raw, SENTIMENT_3WAY)
    assert out["sentiment"] == "positive"


def test_parse_labels_invalid_choice_nulled():
    """单选字段值不在选项里 → 设 None。"""
    raw = '{"sentiment": "mega_bullish"}'
    out = _parse_labels(raw, SENTIMENT_3WAY)
    assert out["sentiment"] is None


def test_parse_labels_missing_field_is_none():
    raw = '{}'
    out = _parse_labels(raw, SENTIMENT_3WAY)
    assert out["sentiment"] is None


def test_parse_labels_with_code_fence():
    raw = '```json\n{"sentiment": "negative"}\n```'
    out = _parse_labels(raw, SENTIMENT_3WAY)
    assert out["sentiment"] == "negative"


def test_parse_labels_list_field():
    scheme = LabelScheme(fields={"companies": "list[str]"})
    out = _parse_labels('{"companies": ["AAPL", "MSFT"]}', scheme)
    assert out["companies"] == ["AAPL", "MSFT"]


def test_parse_labels_list_field_invalid_becomes_empty():
    scheme = LabelScheme(fields={"companies": "list[str]"})
    out = _parse_labels('{"companies": "not a list"}', scheme)
    assert out["companies"] == []


def test_parse_labels_float_field():
    scheme = LabelScheme(fields={"confidence": "float[0-1]"})
    out = _parse_labels('{"confidence": 0.85}', scheme)
    assert out["confidence"] == 0.85


def test_parse_labels_float_invalid_becomes_none():
    scheme = LabelScheme(fields={"confidence": "float[0-1]"})
    out = _parse_labels('{"confidence": "high"}', scheme)
    assert out["confidence"] is None


def test_parse_labels_raises_on_bad_json():
    with pytest.raises(ValueError, match="JSON"):
        _parse_labels("not json at all", SENTIMENT_3WAY)


def test_parse_labels_raises_on_array_root():
    with pytest.raises(ValueError, match="object"):
        _parse_labels("[1, 2, 3]", SENTIMENT_3WAY)


# --- LabelScheme.to_prompt_block ---------------------------------------------

def test_label_scheme_to_prompt_includes_choices():
    block = SENTIMENT_3WAY.to_prompt_block()
    assert "sentiment" in block
    assert "positive" in block
    assert "negative" in block


def test_label_scheme_to_prompt_includes_description():
    scheme = LabelScheme(
        fields={"x": ["a", "b"]},
        description="自定义说明",
    )
    block = scheme.to_prompt_block()
    assert "自定义说明" in block


# --- LLMLabeler ---------------------------------------------------------------

def _make_mock_client(responses) -> LLMClient:
    """responses: list of str（按调用顺序返回）或单个 str。"""
    c = LLMClient(backend="deepseek", api_key="sk-test")
    if isinstance(responses, list):
        c.chat = MagicMock(side_effect=responses)
    else:
        c.chat = MagicMock(return_value=responses)
    return c


def test_labeler_label_one():
    client = _make_mock_client('{"sentiment": "positive"}')
    labeler = LLMLabeler(llm_client=client)
    sample = labeler.label_one("Apple beats earnings", SENTIMENT_3WAY)
    assert isinstance(sample, LabeledSample)
    assert sample.text == "Apple beats earnings"
    assert sample.labels["sentiment"] == "positive"
    assert "llm" in sample.backend


def test_labeler_label_one_raises_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    labeler = LLMLabeler()
    with pytest.raises(LLMNotAvailable):
        labeler.label_one("text", SENTIMENT_3WAY)


def test_labeler_batch_three_texts():
    client = _make_mock_client([
        '{"sentiment": "positive"}',
        '{"sentiment": "negative"}',
        '{"sentiment": "neutral"}',
    ])
    labeler = LLMLabeler(llm_client=client)
    samples = labeler.label_batch(["t1", "t2", "t3"], SENTIMENT_3WAY)
    assert len(samples) == 3
    assert samples[0].labels["sentiment"] == "positive"
    assert samples[1].labels["sentiment"] == "negative"
    assert samples[2].labels["sentiment"] == "neutral"


def test_labeler_batch_skips_errors():
    client = _make_mock_client([
        '{"sentiment": "positive"}',
        'NOT JSON',
        '{"sentiment": "negative"}',
    ])
    labeler = LLMLabeler(llm_client=client)
    samples = labeler.label_batch(["t1", "t2", "t3"], SENTIMENT_3WAY,
                                   skip_errors=True)
    assert len(samples) == 3
    assert samples[1].backend == "error"
    assert samples[0].labels["sentiment"] == "positive"


def test_labeler_batch_raises_when_skip_disabled():
    client = _make_mock_client([
        '{"sentiment": "positive"}',
        'NOT JSON',
    ])
    labeler = LLMLabeler(llm_client=client)
    with pytest.raises(ValueError):
        labeler.label_batch(["t1", "t2"], SENTIMENT_3WAY, skip_errors=False)


def test_labeler_to_dict_serializable():
    import json
    client = _make_mock_client('{"sentiment": "positive"}')
    labeler = LLMLabeler(llm_client=client)
    sample = labeler.label_one("x", SENTIMENT_3WAY)
    d = sample.to_dict()
    assert json.dumps(d, ensure_ascii=False)
    assert d["text"] == "x"


# --- PRESETS -----------------------------------------------------------------

def test_presets_have_sentiment_3way():
    assert "sentiment_3way" in PRESETS
    assert PRESETS["sentiment_3way"] is SENTIMENT_3WAY


def test_presets_event_type_has_required_fields():
    scheme = PRESETS["event_type"]
    assert "sentiment" in scheme.fields
    assert "event_type" in scheme.fields
    assert "affected_companies" in scheme.fields
