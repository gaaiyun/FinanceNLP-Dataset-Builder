> **维护状态说明**：本仓库当前是 AI 辅助生成的初始脚手架，未在生产环境持续打磨。代码可作为参考与起点，使用前请自行核对接口、依赖与边界条件。如果你打算接手维护、把它合并到其他项目，或者发现 bug，欢迎开 issue 或 PR。
# FinanceNLP Dataset Builder

<div align="center">

**金融 NLP 数据集构建工具**

📰 财经新闻采集 | 📄 财报文本解析 | 💬 社交媒体情绪 | 🧹 数据清洗标注 | 📤 多格式导出

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🌟 功能特性

### 数据采集
- ✅ **财经新闻**: 支持 Finnhub、NewsAPI、Yahoo Finance 等多个数据源
- ✅ **财报文档**: SEC EDGAR 财报自动抓取和解析
- ✅ **社交媒体**: Twitter、Reddit 情绪数据采集
- ✅ **多语言支持**: 中文和英文数据采集

### 数据处理
- ✅ **智能去重**: SimHash 算法检测相似文本
- ✅ **文本标准化**: 时间、货币、数字格式统一
- ✅ **情感标注**: FinBERT/规则-based 情感分析
- ✅ **实体识别**: 公司名、人名、金额自动提取
- ✅ **关键词提取**: 金融领域关键词自动识别

### 数据集构建
- ✅ **多格式导出**: JSON、JSONL、CSV、Parquet
- ✅ **数据集划分**: 自动划分 train/val/test
- ✅ **HuggingFace**: 直接导出为 HF Dataset 格式
- ✅ **数据卡片**: 自动生成数据集说明文档

## 🚀 快速开始

### 1. 安装依赖

```bash
cd workspace/skills/finance-nlp-dataset
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# 新闻数据源
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_API_KEY=your_newsapi_key

# 社交媒体
TWITTER_BEARER_TOKEN=your_twitter_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_secret

# 文本处理（可选）
OPENAI_API_KEY=your_openai_key
```

### 3. 基本使用

```python
from finance_nlp_dataset import FinanceNLPBuilder

# 初始化构建器
builder = FinanceNLPBuilder(
    output_dir="./datasets",
    language="en",
    sentiment_model="rules"  # 使用规则-based 情感分析
)

# 采集财经新闻
news_data = builder.collect_news(
    symbols=["AAPL", "TSLA", "NVDA"],
    days=7,
    sources=["finnhub"]
)

# 数据清洗和标注
cleaned_data = builder.clean_and_label(
    data=news_data,
    tasks=["dedup", "normalize", "sentiment"]
)

# 导出数据集
output_path = builder.export_dataset(
    data=cleaned_data,
    filename="finance_news_dataset",
    format="json"
)

print(f"数据集已导出：{output_path}")
```

### 4. 一键构建完整数据集

```python
builder = FinanceNLPBuilder(
    output_dir="./datasets",
    language="en"
)

# 一键完成所有步骤
output_path = builder.build_complete_dataset(
    symbols=["AAPL", "MSFT", "GOOGL"],
    news_days=7,
    social_days=3,
    quarters=["Q4"],
    output_filename="complete_finance_nlp_dataset"
)
```

## 📖 详细文档

### 数据采集

#### 财经新闻采集

```python
# 多源采集
news_data = builder.collect_news(
    symbols=["AAPL", "TSLA"],
    days=7,                    # 采集 7 天内的新闻
    sources=["finnhub", "newsapi", "yahoo"],  # 多个数据源
    categories=["company", "market"]  # 新闻类别
)
```

#### 财报解析

```python
earnings_data = builder.parse_earnings(
    symbols=["AAPL", "MSFT"],
    quarters=["Q1", "Q2", "Q3", "Q4"],
    years=[2023, 2024],
    source="sec_edgar"
)
```

#### 社交媒体情绪

```python
social_data = builder.collect_social_sentiment(
    symbols=["TSLA", "NVDA"],
    platforms=["twitter", "reddit"],
    days=3,
    limit_per_symbol=100
)
```

### 数据处理

#### 自定义处理流程

```python
# 单独调用处理功能
processor = builder.processor

# 去重
deduped = processor.deduplicate(raw_data)

# 标准化
normalized = processor.normalize_text(deduped)

# 情感标注
labeled = processor.add_sentiment_labels(normalized)

# 实体识别
with_entities = processor.extract_entities(labeled)

# 关键词提取
final_data = processor.extract_keywords(with_entities)
```

#### 配置处理参数

```python
builder = FinanceNLPBuilder(
    dedup_threshold=0.85,      # 去重相似度阈值
    min_text_length=50,        # 最小文本长度
    max_text_length=5000,      # 最大文本长度
    sentiment_model="finbert", # finbert/openai/rules
    entity_recognition=True,   # 启用实体识别
    keyword_extraction=True    # 启用关键词提取
)
```

### 数据集导出

#### 多格式导出

```python
# JSON 格式
builder.export_dataset(data, "dataset", format="json")

# JSONL 格式（适合 LLM 训练）
builder.export_dataset(data, "dataset", format="jsonl")

# CSV 格式（适合传统 ML）
builder.export_dataset(data, "dataset", format="csv")

# Parquet 格式（适合大数据）
builder.export_dataset(data, "dataset", format="parquet")

# HuggingFace Dataset
builder.export_dataset(data, "dataset", format="hf_dataset")
```

#### 数据集划分

```python
# 自动划分 train/val/test
builder.export_dataset(
    data=data,
    filename="dataset",
    split={
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    }
)
```

## 📊 输出数据格式

### 新闻数据

```json
{
  "id": "news_001",
  "type": "news",
  "title": "Apple Reports Record Q4 Earnings",
  "content": "Apple Inc. announced record-breaking...",
  "source": "finnhub",
  "symbols": ["AAPL"],
  "published_at": "2024-01-15T10:30:00Z",
  "sentiment": "positive",
  "sentiment_score": 0.85,
  "entities": [
    {"text": "Apple Inc.", "type": "ORGANIZATION"},
    {"text": "Tim Cook", "type": "PERSON"}
  ],
  "keywords": ["earnings", "revenue", "growth", "Apple"],
  "language": "en"
}
```

### 财报数据

```json
{
  "id": "earnings_001",
  "type": "earnings",
  "symbol": "AAPL",
  "quarter": "Q4",
  "year": 2024,
  "filing_date": "2024-01-25",
  "document_text": "Apple Inc. 10-K Annual Report...",
  "sections": {
    "revenue": "Revenue increased by 8%...",
    "risk_factors": "Risk factors include..."
  },
  "metrics": {
    "revenue": 119500000000,
    "net_income": 33900000000
  },
  "language": "en"
}
```

### 社交媒体数据

```json
{
  "id": "social_001",
  "type": "social_media",
  "platform": "twitter",
  "symbol": "TSLA",
  "text": "$TSLA production numbers look amazing! 🚀",
  "author": "investor_joe",
  "posted_at": "2024-01-15T14:22:00Z",
  "sentiment": "positive",
  "sentiment_score": 0.92,
  "engagement": {
    "likes": 1250,
    "retweets": 340,
    "comments": 89
  },
  "language": "en"
}
```

## 🧪 测试

运行单元测试：

```bash
python tests.py
```

运行示例：

```bash
python examples.py
```

## 📁 项目结构

```
finance-nlp-dataset/
├── SKILL.md              # 技能描述
├── __init__.py           # 主模块
├── data_collector.py     # 数据采集
├── data_processor.py     # 数据处理
├── dataset_builder.py    # 数据集构建
├── examples.py           # 使用示例
├── tests.py             # 单元测试
├── requirements.txt      # 依赖包
└── README.md            # 本文档
```

## 🔧 配置选项

### 完整配置示例

```python
builder = FinanceNLPBuilder(
    # 输出配置
    output_dir="./datasets",
    language="zh",
    
    # 数据源配置
    news_sources=["finnhub", "newsapi", "yahoo"],
    social_platforms=["twitter", "reddit"],
    
    # 处理配置
    dedup_threshold=0.85,
    min_text_length=50,
    max_text_length=5000,
    sentiment_model="finbert",
    entity_recognition=True,
    keyword_extraction=True,
    
    # 导出配置
    export_format="parquet",
    compress=True
)
```

## 🎯 使用场景

### 1. 金融情感分析训练数据

```python
# 采集新闻和社交媒体数据
news = builder.collect_news(symbols=["AAPL"], days=30)
social = builder.collect_social_sentiment(symbols=["AAPL"], days=30)

# 合并和标注
all_data = news + social
labeled = builder.clean_and_label(all_data, tasks=["sentiment"])

# 导出
builder.export_dataset(labeled, "sentiment_dataset", split={"train": 0.8, "test": 0.2})
```

### 2. 财经新闻分类数据集

```python
news = builder.collect_news(
    symbols=["AAPL", "GOOGL", "MSFT"],
    days=60,
    categories=["company", "market", "technology"]
)

# 添加关键词和实体
processed = builder.clean_and_label(news, tasks=["entities", "keywords"])

builder.export_dataset(processed, "news_classification_dataset", format="jsonl")
```

### 3. 财报问答数据集

```python
earnings = builder.parse_earnings(
    symbols=["AAPL", "MSFT"],
    quarters=["Q1", "Q2", "Q3", "Q4"],
    years=[2023, 2024]
)

builder.export_dataset(earnings, "earnings_qa_dataset", format="json")
```

## ⚠️ 注意事项

1. **API Keys**: 部分数据源需要 API 密钥，请提前配置
2. **速率限制**: 注意 API 调用频率限制
3. **数据质量**: 建议人工抽检数据质量
4. **存储空间**: 大规模数据集需要充足存储空间
5. **内存使用**: 处理大量数据时注意内存占用

## 📝 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

如有问题，请通过 Issue 或邮件联系。

---

**派蒙提示**: 数据集构建完成记得给派蒙摩拉肉奖励哦~ ⭐🍖
