# FinanceNLP Dataset Builder - 金融 NLP 数据集构建工具

## 技能描述

本技能提供完整的金融 NLP 数据集构建框架，支持：
- 📰 财经新闻数据采集（多源聚合）
- 📄 财报文本解析（PDF/HTML）
- 💬 社交媒体情绪数据（Twitter/Reddit/微博）
- 🧹 数据清洗和标注（去重、标准化、情感标注）
- 📤 数据集导出（JSON/CSV/Parquet 格式）

## 使用场景

- 构建金融情感分析训练数据集
- 创建财经新闻分类数据集
- 准备财报问答（Q&A）数据集
- 收集社交媒体情绪标注数据
- 构建金融事件抽取数据集

## 快速开始

### 1. 安装依赖

```bash
cd workspace/skills/finance-nlp-dataset
pip install -r requirements.txt
```

### 2. 配置 API Keys

在 `.env` 文件中配置必要的 API 密钥：

```env
# 新闻数据源
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_API_KEY=your_newsapi_key

# 社交媒体
TWITTER_BEARER_TOKEN=your_twitter_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_secret

# 文本处理
OPENAI_API_KEY=your_openai_key  # 用于情感标注（可选）
```

### 3. 基本使用

```python
from finance_nlp_dataset import FinanceNLPBuilder

# 初始化构建器
builder = FinanceNLPBuilder(
    output_dir="./datasets",
    language="zh"  # 或 "en"
)

# 采集财经新闻
news_data = builder.collect_news(
    symbols=["AAPL", "TSLA", "NVDA"],
    days=7,
    sources=["finnhub", "newsapi"]
)

# 解析财报文本
earnings_data = builder.parse_earnings(
    symbols=["AAPL"],
    quarters=["Q4 2024"],
    source="sec_edgar"
)

# 采集社交媒体情绪
social_data = builder.collect_social_sentiment(
    symbols=["AAPL", "TSLA"],
    platforms=["twitter", "reddit"],
    days=3
)

# 数据清洗和标注
cleaned_data = builder.clean_and_label(
    data=news_data + earnings_data + social_data,
    tasks=["dedup", "normalize", "sentiment"]
)

# 导出数据集
builder.export_dataset(
    data=cleaned_data,
    format="parquet",  # 或 "json", "csv"
    filename="finance_nlp_dataset_2024"
)
```

## 核心功能

### 数据采集 (DataCollector)

支持多种数据源：
- **财经新闻**: Finnhub, NewsAPI, Yahoo Finance
- **财报文档**: SEC EDGAR, 公司官网
- **社交媒体**: Twitter, Reddit, 微博（可选）
- **宏观经济**: FRED, 央行数据

### 数据处理 (DataProcessor)

提供数据清洗和预处理：
- 文本去重（SimHash/MinHash）
- 标准化（时间、货币、数字）
- 情感标注（正面/负面/中性）
- 实体识别（公司名、人名、地点）
- 关键词提取

### 数据集构建 (DatasetBuilder)

支持多种输出格式：
- JSON/JSONL（适合 LLM 训练）
- CSV（适合传统 ML）
- Parquet（适合大数据处理）
- HuggingFace Dataset（直接上传）

## 输出数据集格式

### 新闻数据集
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
  "entities": ["Apple Inc.", "Tim Cook"],
  "language": "en"
}
```

### 财报数据集
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
    "risk_factors": "Risk factors include...",
    "md&a": "Management's Discussion..."
  },
  "metrics": {
    "revenue": 119500000000,
    "net_income": 33900000000
  }
}
```

### 社交媒体情绪数据集
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
  }
}
```

## 配置选项

```python
builder = FinanceNLPBuilder(
    # 输出配置
    output_dir="./datasets",
    language="zh",  # zh/en
    
    # 数据源配置
    news_sources=["finnhub", "newsapi", "yahoo"],
    social_platforms=["twitter", "reddit"],
    
    # 处理配置
    dedup_threshold=0.85,  # 去重相似度阈值
    min_text_length=50,    # 最小文本长度
    max_text_length=5000,  # 最大文本长度
    
    # 标注配置
    sentiment_model="finbert",  # finbert/openai/zero-shot
    entity_recognition=True,
    keyword_extraction=True,
    
    # 导出配置
    export_format="parquet",
    compress=True
)
```

## 依赖

见 `requirements.txt`

## 测试

```bash
python -m pytest tests/
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

---

本工具专为金融 NLP 研究设计，用于构建训练数据集。
