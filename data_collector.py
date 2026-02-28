#!/usr/bin/env python3
"""
数据采集模块

支持多种数据源：
- 财经新闻：Finnhub, NewsAPI, Yahoo Finance
- 财报文档：SEC EDGAR
- 社交媒体：Twitter, Reddit
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataCollector:
    """数据采集器"""
    
    def __init__(
        self,
        language: str = "zh",
        news_sources: List[str] = None,
        social_platforms: List[str] = None
    ):
        """
        初始化数据采集器
        
        Args:
            language: 语言 (zh/en)
            news_sources: 新闻数据源列表
            social_platforms: 社交媒体平台列表
        """
        self.language = language
        self.news_sources = news_sources or ["finnhub", "newsapi"]
        self.social_platforms = social_platforms or ["twitter", "reddit"]
        
        # 加载 API Keys
        self.api_keys = self._load_api_keys()
        
        logger.info(f"DataCollector 初始化完成 - 语言：{language}")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """从环境变量加载 API Keys"""
        return {
            "finnhub": os.getenv("FINNHUB_API_KEY", ""),
            "newsapi": os.getenv("NEWSAPI_API_KEY", ""),
            "twitter": os.getenv("TWITTER_BEARER_TOKEN", ""),
            "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", ""),
            "reddit_secret": os.getenv("REDDIT_SECRET", ""),
            "openai": os.getenv("OPENAI_API_KEY", ""),
        }
    
    def collect_financial_news(
        self,
        symbols: List[str],
        days: int = 7,
        source: str = "finnhub",
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        采集财经新闻
        
        Args:
            symbols: 股票代码列表
            days: 采集天数
            source: 数据源
            categories: 新闻类别
            
        Returns:
            新闻数据列表
        """
        if source == "finnhub":
            return self._collect_finnhub_news(symbols, days, categories)
        elif source == "newsapi":
            return self._collect_newsapi_news(symbols, days, categories)
        elif source == "yahoo":
            return self._collect_yahoo_news(symbols, days)
        else:
            logger.warning(f"不支持的新闻源：{source}")
            return []
    
    def _collect_finnhub_news(
        self,
        symbols: List[str],
        days: int,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """从 Finnhub 采集新闻"""
        if not self.api_keys["finnhub"]:
            logger.warning("Finnhub API Key 未配置")
            return []
        
        all_news = []
        base_url = "https://finnhub.io/api/v1/company-news"
        
        for symbol in symbols:
            try:
                # 计算日期范围
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                params = {
                    "symbol": symbol,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "token": self.api_keys["finnhub"]
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                news_items = response.json()
                
                for item in news_items:
                    news = {
                        "id": f"finnhub_{symbol}_{item.get('id', '')}",
                        "type": "news",
                        "title": item.get("headline", ""),
                        "content": item.get("summary", ""),
                        "source": "finnhub",
                        "symbols": [symbol],
                        "published_at": datetime.fromtimestamp(item.get("datetime", 0)).isoformat(),
                        "url": item.get("url", ""),
                        "category": item.get("category", ""),
                        "language": self.language
                    }
                    all_news.append(news)
                
                logger.info(f"从 Finnhub 采集 {symbol} 新闻 {len(news_items)} 条")
                
            except Exception as e:
                logger.error(f"采集 Finnhub 新闻失败 ({symbol}): {str(e)}")
        
        return all_news
    
    def _collect_newsapi_news(
        self,
        symbols: List[str],
        days: int,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """从 NewsAPI 采集新闻"""
        if not self.api_keys["newsapi"]:
            logger.warning("NewsAPI Key 未配置")
            return []
        
        all_news = []
        base_url = "https://newsapi.org/v2/everything"
        
        for symbol in symbols:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # 构建搜索查询
                q = f'"{symbol}" AND (stock OR earnings OR finance OR market)'
                
                params = {
                    "q": q,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "language": "en" if self.language == "en" else "zh",
                    "sortBy": "publishedAt",
                    "apiKey": self.api_keys["newsapi"]
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                articles = data.get("articles", [])
                
                for article in articles:
                    news = {
                        "id": f"newsapi_{symbol}_{article.get('title', '')[:20]}",
                        "type": "news",
                        "title": article.get("title", ""),
                        "content": article.get("description", "") + " " + article.get("content", ""),
                        "source": "newsapi",
                        "symbols": [symbol],
                        "published_at": article.get("publishedAt", ""),
                        "url": article.get("url", ""),
                        "author": article.get("author", ""),
                        "language": self.language
                    }
                    all_news.append(news)
                
                logger.info(f"从 NewsAPI 采集 {symbol} 新闻 {len(articles)} 条")
                
            except Exception as e:
                logger.error(f"采集 NewsAPI 新闻失败 ({symbol}): {str(e)}")
        
        return all_news
    
    def _collect_yahoo_news(
        self,
        symbols: List[str],
        days: int
    ) -> List[Dict[str, Any]]:
        """从 Yahoo Finance 采集新闻（使用 yfinance 库）"""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance 未安装，跳过 Yahoo Finance 新闻采集")
            return []
        
        all_news = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                news_items = stock.news
                
                for item in news_items:
                    # 检查日期
                    pub_time = datetime.fromtimestamp(item.get("providerPublishTime", 0))
                    if (datetime.now() - pub_time).days > days:
                        continue
                    
                    news = {
                        "id": f"yahoo_{symbol}_{item.get('uuid', '')}",
                        "type": "news",
                        "title": item.get("title", ""),
                        "content": item.get("summary", ""),
                        "source": "yahoo",
                        "symbols": [symbol],
                        "published_at": pub_time.isoformat(),
                        "url": item.get("link", ""),
                        "publisher": item.get("publisher", ""),
                        "language": self.language
                    }
                    all_news.append(news)
                
                logger.info(f"从 Yahoo Finance 采集 {symbol} 新闻 {len(news_items)} 条")
                
            except Exception as e:
                logger.error(f"采集 Yahoo Finance 新闻失败 ({symbol}): {str(e)}")
        
        return all_news
    
    def collect_earnings_reports(
        self,
        symbols: List[str],
        quarters: List[str] = None,
        years: List[int] = None,
        source: str = "sec_edgar"
    ) -> List[Dict[str, Any]]:
        """
        采集财报报告
        
        Args:
            symbols: 股票代码列表
            quarters: 季度列表
            years: 年份列表
            source: 数据源
            
        Returns:
            财报数据列表
        """
        if source == "sec_edgar":
            return self._collect_sec_filings(symbols, quarters, years)
        else:
            logger.warning(f"不支持的财报源：{source}")
            return []
    
    def _collect_sec_filings(
        self,
        symbols: List[str],
        quarters: List[str] = None,
        years: List[int] = None
    ) -> List[Dict[str, Any]]:
        """从 SEC EDGAR 采集财报"""
        if quarters is None:
            quarters = ["Q1", "Q2", "Q3", "Q4"]
        if years is None:
            years = [datetime.now().year, datetime.now().year - 1]
        
        all_earnings = []
        
        for symbol in symbols:
            for year in years:
                for quarter in quarters:
                    try:
                        # 简化实现：实际应该调用 SEC EDGAR API
                        earning = {
                            "id": f"sec_{symbol}_{year}_{quarter}",
                            "type": "earnings",
                            "symbol": symbol,
                            "quarter": quarter,
                            "year": year,
                            "filing_date": f"{year}-{self._quarter_to_month(quarter)}-15",
                            "document_text": f"[{symbol} {year} {quarter} Earnings Report]",
                            "sections": {},
                            "metrics": {},
                            "source": "sec_edgar",
                            "language": "en"
                        }
                        all_earnings.append(earning)
                        
                    except Exception as e:
                        logger.error(f"采集 SEC 财报失败 ({symbol} {year} {quarter}): {str(e)}")
        
        return all_earnings
    
    def _quarter_to_month(self, quarter: str) -> str:
        """季度转月份"""
        mapping = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}
        return mapping.get(quarter, "03")
    
    def collect_social_media_sentiment(
        self,
        symbols: List[str],
        days: int = 3,
        platform: str = "twitter",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        采集社交媒体情绪数据
        
        Args:
            symbols: 股票代码列表
            days: 采集天数
            platform: 平台
            limit: 每个股票限制数量
            
        Returns:
            社交媒体数据列表
        """
        if platform == "twitter":
            return self._collect_twitter_sentiment(symbols, days, limit)
        elif platform == "reddit":
            return self._collect_reddit_sentiment(symbols, days, limit)
        else:
            logger.warning(f"不支持的社交平台：{platform}")
            return []
    
    def _collect_twitter_sentiment(
        self,
        symbols: List[str],
        days: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """从 Twitter 采集情绪数据"""
        if not self.api_keys["twitter"]:
            logger.warning("Twitter API Key 未配置")
            return []
        
        all_tweets = []
        
        for symbol in symbols:
            try:
                # Twitter API v2 端点
                url = f"https://api.twitter.com/2/tweets/search/recent"
                
                # 构建查询
                query = f"${symbol} OR #{symbol} OR {symbol} stock -is:retweet"
                
                params = {
                    "query": query,
                    "max_results": min(limit, 100),
                    "tweet.fields": "created_at,public_metrics,author_id",
                    "expansions": "author_id"
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_keys['twitter']}"
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    tweets = data.get("data", [])
                    
                    for tweet in tweets:
                        social = {
                            "id": f"twitter_{tweet.get('id', '')}",
                            "type": "social_media",
                            "platform": "twitter",
                            "symbol": symbol,
                            "text": tweet.get("text", ""),
                            "author": tweet.get("author_id", ""),
                            "posted_at": tweet.get("created_at", ""),
                            "engagement": tweet.get("public_metrics", {}),
                            "language": self.language
                        }
                        all_tweets.append(social)
                    
                    logger.info(f"从 Twitter 采集 {symbol} 数据 {len(tweets)} 条")
                else:
                    logger.warning(f"Twitter API 请求失败：{response.status_code}")
                    
            except Exception as e:
                logger.error(f"采集 Twitter 数据失败 ({symbol}): {str(e)}")
        
        return all_tweets
    
    def _collect_reddit_sentiment(
        self,
        symbols: List[str],
        days: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """从 Reddit 采集情绪数据"""
        if not self.api_keys["reddit_client_id"]:
            logger.warning("Reddit API Keys 未配置")
            return []
        
        all_posts = []
        
        for symbol in symbols:
            try:
                # Reddit OAuth 认证
                auth = requests.auth.HTTPBasicAuth(
                    self.api_keys["reddit_client_id"],
                    self.api_keys["reddit_secret"]
                )
                
                # 获取 access token
                oauth_response = requests.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=auth,
                    data={"grant_type": "client_credentials"},
                    headers={"User-Agent": "FinanceNLP/1.0"}
                )
                
                if oauth_response.status_code != 200:
                    logger.warning("Reddit OAuth 失败")
                    continue
                
                token = oauth_response.json()["access_token"]
                headers = {"Authorization": f"Bearer {token}", "User-Agent": "FinanceNLP/1.0"}
                
                # 搜索相关 subreddit
                subreddits = ["wallstreetbets", "stocks", "investing", "finance"]
                
                for subreddit in subreddits:
                    url = f"https://oauth.reddit.com/r/{subreddit}/search"
                    params = {
                        "q": symbol,
                        "limit": min(limit // len(subreddits), 50),
                        "sort": "new",
                        "t": f"{days}d"
                    }
                    
                    response = requests.get(url, params=params, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get("data", {}).get("children", [])
                        
                        for post in posts:
                            post_data = post.get("data", {})
                            social = {
                                "id": f"reddit_{post_data.get('id', '')}",
                                "type": "social_media",
                                "platform": "reddit",
                                "symbol": symbol,
                                "text": post_data.get("title", "") + " " + post_data.get("selftext", ""),
                                "author": post_data.get("author", ""),
                                "posted_at": datetime.fromtimestamp(post_data.get("created_utc", 0)).isoformat(),
                                "engagement": {
                                    "upvotes": post_data.get("ups", 0),
                                    "comments": post_data.get("num_comments", 0)
                                },
                                "subreddit": subreddit,
                                "language": "en"
                            }
                            all_posts.append(social)
                        
                        logger.info(f"从 Reddit r/{subreddit} 采集 {symbol} 数据 {len(posts)} 条")
                    
            except Exception as e:
                logger.error(f"采集 Reddit 数据失败 ({symbol}): {str(e)}")
        
        return all_posts
