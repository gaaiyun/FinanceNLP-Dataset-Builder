"""free_sources.py 测试 —— 不发网络。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.free_sources import (
    NewsSample,
    _strip_cdata,
    _strip_tags,
    fetch_rss,
    fetch_yfinance_news,
    synthetic_samples,
)


# --- synthetic ---------------------------------------------------------------

def test_synthetic_samples_default():
    items = synthetic_samples()
    assert len(items) == 6
    assert all(isinstance(s, NewsSample) for s in items)
    assert all(s.source == "synthetic" for s in items)


def test_synthetic_samples_deterministic():
    a = synthetic_samples(n=10, seed=42)
    b = synthetic_samples(n=10, seed=42)
    assert [s.title for s in a] == [s.title for s in b]


def test_synthetic_samples_custom_count():
    items = synthetic_samples(n=3)
    assert len(items) == 3


def test_news_sample_to_dict():
    s = NewsSample(title="t", text="body", source="synthetic", url="http://x",
                   published_at="2024-01-01", ticker="AAPL")
    d = s.to_dict()
    assert d["title"] == "t"
    assert d["source"] == "synthetic"
    assert d["ticker"] == "AAPL"


# --- yfinance（mock）--------------------------------------------------------

def test_fetch_yfinance_news_parses_items():
    fake_yf_news = [
        {"title": "Apple Q3 Earnings", "link": "http://x.com/1",
         "publisher": "Reuters", "providerPublishTime": 1700000000},
        {"title": "iPhone Sales", "link": "http://x.com/2",
         "publisher": "Bloomberg", "providerPublishTime": 1700086400},
    ]
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value.news = fake_yf_news
    with patch.dict("sys.modules", {"yfinance": mock_yf}):
        items = fetch_yfinance_news("AAPL", max_items=5)
    assert len(items) == 2
    assert items[0].title == "Apple Q3 Earnings"
    assert items[0].ticker == "AAPL"
    assert items[0].source.startswith("yfinance:")


def test_fetch_yfinance_skips_items_without_title():
    fake_yf_news = [
        {"title": "Valid News", "providerPublishTime": 1700000000},
        {"link": "http://x.com/missing-title"},   # 没 title 跳过
    ]
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value.news = fake_yf_news
    with patch.dict("sys.modules", {"yfinance": mock_yf}):
        items = fetch_yfinance_news("AAPL")
    assert len(items) == 1
    assert items[0].title == "Valid News"


def test_fetch_yfinance_news_raises_when_not_installed(monkeypatch):
    import builtins
    real = builtins.__import__
    def fake(name, *a, **kw):
        if name == "yfinance":
            raise ImportError("simulated")
        return real(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="yfinance"):
        fetch_yfinance_news("AAPL")


def test_fetch_yfinance_max_items_caps_count():
    fake = [{"title": f"News {i}", "providerPublishTime": 1700000000 + i}
            for i in range(50)]
    mock_yf = MagicMock()
    mock_yf.Ticker.return_value.news = fake
    with patch.dict("sys.modules", {"yfinance": mock_yf}):
        items = fetch_yfinance_news("AAPL", max_items=7)
    assert len(items) == 7


# --- RSS ---------------------------------------------------------------------

_SAMPLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Test Feed</title>
<item>
  <title><![CDATA[Apple beats earnings]]></title>
  <description>Apple Inc reported strong Q3 results.</description>
  <link>https://example.com/article-1</link>
  <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
</item>
<item>
  <title>Tesla announces buyback</title>
  <description><![CDATA[<p>Tesla plans $7.5B buyback program.</p>]]></description>
  <link>https://example.com/article-2</link>
</item>
</channel>
</rss>"""


def test_fetch_rss_parses_items(monkeypatch):
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return _SAMPLE_RSS

    monkeypatch.setattr(
        "scripts.free_sources.urllib.request.urlopen",
        lambda *a, **kw: FakeResp(),
    )
    items = fetch_rss("http://test.com/feed.xml")
    assert len(items) == 2
    assert items[0].title == "Apple beats earnings"
    # 第二条的 <p> 标签应该被剥掉
    assert "<p>" not in items[1].text
    assert items[0].source.startswith("rss:")


def test_fetch_rss_no_items_raises(monkeypatch):
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"<?xml?><html><body>not rss</body></html>"

    monkeypatch.setattr(
        "scripts.free_sources.urllib.request.urlopen",
        lambda *a, **kw: FakeResp(),
    )
    with pytest.raises(RuntimeError, match="<item>"):
        fetch_rss("http://test.com/feed")


# --- _strip helpers ----------------------------------------------------------

def test_strip_cdata():
    assert _strip_cdata("<![CDATA[hello]]>") == "hello"


def test_strip_cdata_no_cdata():
    assert _strip_cdata("plain text") == "plain text"


def test_strip_tags():
    assert _strip_tags("<p>hello <b>world</b></p>") == "hello world"


def test_strip_tags_empty():
    assert _strip_tags("") == ""
