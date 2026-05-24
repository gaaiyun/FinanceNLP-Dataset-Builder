# FinanceNLP-Dataset-Builder

金融 NLP 数据集构建工具：免 API key 数据源 + LLM 自定义标签 + 现有 v1 处理管线。

v1 已有相对完整的采集（Finnhub / NewsAPI / SEC EDGAR）、处理（SimHash 去重、
规则情感、实体识别）、构建（JSONL / Parquet / HuggingFace）三层模块。v2 在
不破坏 v1 的前提下加了两件**开箱可用**的东西：

1. **免 API key 的数据源**：v1 几个源全都要付费 key，对教学 / demo / 个人玩家
   不友好。v2 加 yfinance news（免费、无 key、自带限流）+ 通用 RSS 解析（不依赖
   `feedparser` 包）+ 合成数据。
2. **LLM 自定义标签**：v1 内置 FinBERT 三分类情感是固定 schema。v2 允许用户
   用一句话定义标签字段（"标 sentiment + event_type + 涉及公司"），LLM 按 schema
   一次返回结构化 JSON。

## v2 新增模块

| 模块 | 干什么 |
|---|---|
| `scripts/free_sources.py` | `fetch_yfinance_news(ticker)` / `fetch_rss(url)` / `synthetic_samples(n)`：三个免 key 数据源，统一返回 `NewsSample` |
| `scripts/llm_labeler.py` | `LabelScheme` 自定义字段方案 + `LLMLabeler` 调 LLM 标 + JSON 解析与 schema 校验。内置 `sentiment_3way` / `event_type` 两个 preset |
| `__main__.py` | CLI：`collect` / `label` / `pipeline` / `list-presets` |
| `tests/test_*.py` | 42 个新测试，全部 mock，70ms 跑完 |

## v1 仍保留（不动）

| 模块 | 干什么 |
|---|---|
| `data_collector.py` | Finnhub / NewsAPI / Yahoo Finance / SEC EDGAR / Twitter / Reddit 采集 |
| `data_processor.py` | SimHash 去重、规则情感、命名实体识别、关键词 |
| `dataset_builder.py` | JSONL / Parquet / HuggingFace 导出，train/val/test 切分 |
| `__init__.py` | 顶层 `FinanceNLPBuilder` 一体化入口 |
| `examples.py` / `quickstart.py` / `test_simple.py` | v1 示例和冒烟测试 |

## 安装

```bash
pip install -r requirements.txt

# 可选：v2 用 yfinance 数据源
pip install yfinance

# 可选：v2 用 LLM 标注
pip install openai      # openai / deepseek
pip install anthropic
```

## 快速开始

### v2 入口：免 key 数据 + LLM 标注

```bash
# 1. 列内置标签 preset
python __main__.py list-presets

# 2. 拉合成数据看看格式
python __main__.py collect --source synthetic --max 5 -o data.jsonl

# 3. 从 yfinance 抓真实新闻
python __main__.py collect --source yfinance --ticker AAPL --max 10 -o aapl.jsonl

# 4. 从 RSS 抓
python __main__.py collect --source rss \
    --url https://feeds.bloomberg.com/markets/news.rss --max 20 -o bloomberg.jsonl

# 5. 用 LLM 给 jsonl 数据按 preset 打标签（需 DEEPSEEK_API_KEY）
python __main__.py label --input aapl.jsonl --preset sentiment_3way \
    --backend deepseek -o labeled.jsonl

# 6. 自定义标签 schema
python __main__.py label --input aapl.jsonl \
    --schema-json '{"sentiment":["positive","negative","neutral"],"event_type":["earnings","M&A","other"],"impact_score":"float[0-1]"}' \
    --schema-desc "金融新闻情绪 + 事件类型 + 影响力分数" \
    -o labeled.jsonl

# 7. 一条龙：collect → label
python __main__.py pipeline --source synthetic --max 6 \
    --preset event_type --backend deepseek -o dataset.jsonl
```

### 库调用

```python
# 免 key 拉数据
from scripts.free_sources import fetch_yfinance_news, fetch_rss

news = fetch_yfinance_news("AAPL", max_items=20)
rss_items = fetch_rss("https://feeds.bloomberg.com/markets/news.rss")

# LLM 标注
from scripts.llm_labeler import LLMClient, LLMLabeler, LabelScheme

scheme = LabelScheme(
    fields={
        "sentiment": ["positive", "negative", "neutral"],
        "affected_companies": "list[str]",
        "impact_score": "float[0-1]",
    },
    description="金融新闻三维标注",
)

labeler = LLMLabeler(LLMClient(backend="deepseek"))
samples = labeler.label_batch([n.text for n in news], scheme)
for s in samples:
    print(s.labels)
# {'sentiment': 'positive', 'affected_companies': ['AAPL'], 'impact_score': 0.7}
```

### v1 入口（仍然能用）

```python
from __init__ import FinanceNLPBuilder

builder = FinanceNLPBuilder(language="en", sentiment_model="finbert")
# ... 走 v1 的 collect/process/build 全流程
```

## 内置标签 preset

| 名称 | 字段 |
|---|---|
| `sentiment_3way` | `sentiment`: positive / negative / neutral |
| `event_type` | `sentiment` + `event_type` (earnings/M&A/regulatory/product/macro/other) + `affected_companies: list[str]` |

自定义任何字段方案：传 `--schema-json` 给 CLI 或 `LabelScheme(fields={...})` 给库。
支持单选（列表）、列表字段（`"list[str]"`）、浮点（`"float[0-1]"`）、整数（`"int"`）。

## 设计取舍

- **LLM 缺 key 直接 raise**：v2 的 LLMLabeler 强依赖 LLM；想要 fallback 用
  v1 的 `data_processor.DataProcessor` 走规则标注。**不静默退化** —— 否则用户
  以为拿到的是 LLM 高质量标签，实际是规则法。
- **RSS 解析不依赖 feedparser**：用 stdlib + 正则，少一个依赖。够大多数 RSS 用，
  Atom feed 不一定能解析。
- **yfinance 字段名兼容**：yfinance 不同版本的 `.news` 返回字段名（title vs
  headline，link vs url）有差异，做了兼容。
- **批量标注串行**：v2 的 `label_batch` 不内置并发 —— 不同 LLM 服务的速率限制
  千差万别，并发策略让调用方决定。
- **JSON 解析 best-effort**：单选字段值不在选项里 → 设 None，不抛错；列表字段
  不是 list → 设空列表。这样小批量里有一两条 LLM 跑歪也不影响主流程。

## 项目结构

```
FinanceNLP-Dataset-Builder/
├── __main__.py                    # v2 CLI 统一入口
├── __init__.py                    # v1 顶层 API（FinanceNLPBuilder）
├── scripts/                       # v2 模块
│   ├── free_sources.py            # 免 key 数据源（yfinance/rss/synthetic）
│   └── llm_labeler.py             # LLM 自定义标签
├── data_collector.py              # v1：6 个付费数据源
├── data_processor.py              # v1：去重 + 情感 + NER + 关键词
├── dataset_builder.py             # v1：导出 JSONL/Parquet/HF
├── examples.py                    # v1：用法示例
├── quickstart.py                  # v1：5 分钟上手
├── test_simple.py                 # v1：冒烟测试（imports / 模块结构）
├── tests/                         # v2 pytest（42 个，全 mock）
│   ├── test_free_sources.py
│   └── test_llm_labeler.py
├── requirements.txt
├── .env.example
└── README.md
```

## 测试

```bash
# v2 测试（mock 干净，CI 友好）
pytest tests/ --no-cov

# v1 冒烟
python test_simple.py
```

`tests.py`（v1 留下来的，import 路径假设包名 `finance_nlp_dataset` 但本仓库
目录名含连字符，是错的）已知不可运行 —— 实际效用被 `test_simple.py` 替代。

## 已知限制

- `fetch_yfinance_news` 调的是 Yahoo Finance 内部 API，没正式 SLA，可能被限流。
- `fetch_rss` 不支持 Atom feed（用 `<entry>` 而非 `<item>` 的 feed）。
- LLM 标注成本：每条样本一次调用，大规模标注（>10k 条）账单会上来。
- LLM 输出偶尔不合规（不是有效 JSON），`label_batch(skip_errors=True)` 会把这些
  样本的 labels 全填 None，需要自己再过一遍。

## 许可

MIT
