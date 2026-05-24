"""免 API key 的金融新闻数据源。

v1 的 ``data_collector.py`` 接 Finnhub / NewsAPI / Twitter / Reddit，全部要付费
key。v2 加几个**免 key** 的源，让用户开箱能凑出小样本数据集做 demo / 单测 / 课堂演示：

- ``fetch_yfinance_news(ticker)`` —— yfinance 的 news API（免费、无 key、有限流）
- ``fetch_rss(url)`` —— 通用 RSS 解析（Bloomberg / Reuters / CNBC 都有 RSS）
- ``synthetic_samples()`` —— 合成金融新闻样本，不联网，给测试用

所有源统一返回 ``NewsSample`` dataclass，下游 LLMLabeler 和现有
``data_processor.DataProcessor`` 都能吃。
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class NewsSample:
    title: str
    text: str
    source: str            # "yfinance" / "rss:<host>" / "synthetic"
    url: Optional[str] = None
    published_at: Optional[str] = None    # ISO8601 字符串
    ticker: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at,
            "ticker": self.ticker,
        }


def fetch_yfinance_news(ticker: str, max_items: int = 20) -> List[NewsSample]:
    """从 yfinance 拉某 ticker 的新闻条目。

    yfinance 的 .news 接口是公开免费的（实际上是 Yahoo Finance 内部 API），
    每条返回 title / link / publisher / providerPublishTime。

    Raises
    ------
    ImportError : yfinance 未装
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance 未装。pip install yfinance") from e

    t = yf.Ticker(ticker)
    raw = t.news or []
    out = []
    for item in raw[:max_items]:
        # yfinance 字段名在不同版本里有差异，做兼容
        title = item.get("title") or item.get("headline") or ""
        link = item.get("link") or item.get("url")
        publisher = item.get("publisher") or item.get("source") or "Yahoo"
        ts = item.get("providerPublishTime") or item.get("pubDate")
        published = (datetime.fromtimestamp(ts).isoformat()
                     if isinstance(ts, (int, float)) else
                     (str(ts) if ts else None))
        if not title:
            continue
        out.append(NewsSample(
            title=title,
            text=title,  # yfinance 不返回正文
            source=f"yfinance:{publisher}",
            url=link, published_at=published, ticker=ticker.upper(),
        ))
    return out


# --- RSS 解析（不依赖 feedparser）---------------------------------------------

_ITEM_PAT = re.compile(
    r"<item[^>]*>(.*?)</item>", re.IGNORECASE | re.DOTALL)
_TAG_PAT = re.compile(r"<([a-zA-Z:]+)[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)
_CDATA_PAT = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.IGNORECASE | re.DOTALL)


def _strip_cdata(text: str) -> str:
    return _CDATA_PAT.sub(r"\1", text or "").strip()


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def fetch_rss(url: str, max_items: int = 20, timeout: float = 15.0
              ) -> List[NewsSample]:
    """通用 RSS 解析：取每个 <item> 的 title / description / link / pubDate。

    不依赖 feedparser（pip 包），用 stdlib urllib + 正则。够日常 RSS 用。

    Raises
    ------
    RuntimeError : HTTP 失败 / 返回不是 RSS
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "FinanceNLP-Dataset-Builder/2.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    items_raw = _ITEM_PAT.findall(body)
    if not items_raw:
        raise RuntimeError(f"RSS 没解析出任何 <item>：{url}")

    host = urllib.parse.urlparse(url).hostname or "rss"
    out = []
    for item_block in items_raw[:max_items]:
        fields: dict = {}
        for tag, content in _TAG_PAT.findall(item_block):
            # 先剥 CDATA 再剥 HTML 标签 —— 否则 <![CDATA[...]]> 整体会被
            # _strip_tags 当作 HTML 标签直接吃掉
            fields.setdefault(tag.lower(), _strip_tags(_strip_cdata(content)))
        title = fields.get("title", "")
        desc = fields.get("description") or fields.get("summary") or ""
        link = fields.get("link") or fields.get("guid")
        pub = fields.get("pubdate") or fields.get("published") or fields.get("dc:date")
        if not title:
            continue
        out.append(NewsSample(
            title=title, text=desc or title,
            source=f"rss:{host}",
            url=link, published_at=pub,
        ))
    return out


# --- 合成（测试用）-----------------------------------------------------------

_SYNTH_TEMPLATES = [
    ("Apple beats Q3 estimates with record iPhone sales",
     "Apple Inc. (AAPL) reported Q3 revenue of $89.5B, exceeding analyst "
     "estimates of $87.2B, driven by stronger-than-expected iPhone demand."),
    ("Tesla announces $7.5B share buyback program",
     "Tesla (TSLA) said it will repurchase up to $7.5B of common stock over "
     "the next 18 months, citing strong cash flow."),
    ("SEC opens probe into crypto exchange Binance",
     "The U.S. SEC has launched an investigation into Binance over potential "
     "securities violations, sending crypto markets lower."),
    ("Microsoft completes $69B Activision Blizzard acquisition",
     "Microsoft (MSFT) finalized its $68.7B acquisition of Activision Blizzard "
     "after clearing antitrust hurdles in the UK and EU."),
    ("Fed signals pause on rate hikes amid cooling inflation",
     "Federal Reserve officials indicated they may pause rate increases at the "
     "next FOMC meeting following weaker-than-expected CPI data."),
    ("Goldman Sachs slashes 8% of workforce in restructuring",
     "Goldman Sachs (GS) is cutting approximately 3,200 jobs as part of a "
     "broader cost-reduction initiative, citing slowing dealmaking activity."),
]


def synthetic_samples(n: int = 6, seed: int = 42) -> List[NewsSample]:
    """合成新闻样本，给测试 / 离线 demo 用。"""
    import random
    rng = random.Random(seed)
    base = list(_SYNTH_TEMPLATES)
    out = []
    while len(out) < n:
        title, text = rng.choice(base)
        out.append(NewsSample(
            title=title, text=text, source="synthetic",
            url=None,
            published_at=datetime(2024, 1, 1 + len(out) % 28).isoformat(),
        ))
    return out
