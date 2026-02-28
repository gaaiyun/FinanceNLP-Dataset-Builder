#!/usr/bin/env python3
"""
数据处理模块

提供数据清洗、标准化、标注功能：
- 文本去重（SimHash/MinHash）
- 标准化（时间、货币、数字）
- 情感标注
- 实体识别
- 关键词提取
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""
    
    def __init__(
        self,
        language: str = "zh",
        dedup_threshold: float = 0.85,
        min_text_length: int = 50,
        max_text_length: int = 5000,
        sentiment_model: str = "finbert",
        entity_recognition: bool = True,
        keyword_extraction: bool = True
    ):
        """
        初始化数据处理器
        
        Args:
            language: 语言 (zh/en)
            dedup_threshold: 去重相似度阈值
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
            sentiment_model: 情感分析模型
            entity_recognition: 是否进行实体识别
            keyword_extraction: 是否提取关键词
        """
        self.language = language
        self.dedup_threshold = dedup_threshold
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.sentiment_model = sentiment_model
        self.entity_recognition = entity_recognition
        self.keyword_extraction = keyword_extraction
        
        # 加载模型（延迟加载）
        self._sentiment_pipeline = None
        self._ner_pipeline = None
        
        logger.info(f"DataProcessor 初始化完成 - 语言：{language}")
    
    def deduplicate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        数据去重
        
        使用 SimHash 算法检测相似文本
        
        Args:
            data: 原始数据列表
            
        Returns:
            去重后的数据列表
        """
        logger.info(f"开始去重 - 原始数据量：{len(data)}")
        
        seen_hashes = {}
        unique_data = []
        duplicate_count = 0
        
        for item in data:
            # 提取文本内容
            text = self._extract_text(item)
            if not text:
                unique_data.append(item)
                continue
            
            # 计算 SimHash
            text_hash = self._simhash(text)
            
            # 检查是否重复
            is_duplicate = False
            for existing_hash in seen_hashes.keys():
                similarity = self._hash_similarity(text_hash, existing_hash)
                if similarity > self.dedup_threshold:
                    is_duplicate = True
                    duplicate_count += 1
                    break
            
            if not is_duplicate:
                seen_hashes[text_hash] = item
                unique_data.append(item)
        
        logger.info(f"去重完成 - 移除重复：{duplicate_count} 条，剩余：{len(unique_data)} 条")
        return unique_data
    
    def _extract_text(self, item: Dict[str, Any]) -> str:
        """从数据项中提取文本"""
        if "content" in item:
            return item.get("content", "")
        elif "text" in item:
            return item.get("text", "")
        elif "title" in item:
            return item.get("title", "") + " " + item.get("content", "")
        elif "document_text" in item:
            return item.get("document_text", "")
        return ""
    
    def _simhash(self, text: str) -> int:
        """
        计算 SimHash
        
        简化实现：使用分词 + 哈希
        """
        # 分词（简化：按字符/单词分割）
        if self.language == "zh":
            # 中文：按字符分割
            tokens = list(text.lower())
        else:
            # 英文：按单词分割
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # 计算哈希向量
        vector = [0] * 64
        
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(64):
                if h & (1 << i):
                    vector[i] += 1
                else:
                    vector[i] -= 1
        
        # 生成 SimHash
        simhash = 0
        for i in range(64):
            if vector[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def _hash_similarity(self, hash1: int, hash2: int) -> float:
        """计算两个 SimHash 的相似度"""
        xor = hash1 ^ hash2
        diff = bin(xor).count('1')
        return 1 - (diff / 64)
    
    def normalize_text(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        文本标准化
        
        Args:
            data: 原始数据列表
            
        Returns:
            标准化后的数据列表
        """
        logger.info(f"开始文本标准化 - 数据量：{len(data)}")
        
        normalized_data = []
        
        for item in data:
            normalized_item = item.copy()
            
            # 提取并标准化文本
            for key in ["content", "text", "title"]:
                if key in normalized_item:
                    text = normalized_item[key]
                    if isinstance(text, str):
                        normalized_item[key] = self._standardize_text(text)
            
            # 标准化时间格式
            if "published_at" in normalized_item:
                normalized_item["published_at"] = self._normalize_timestamp(normalized_item["published_at"])
            
            if "posted_at" in normalized_item:
                normalized_item["posted_at"] = self._normalize_timestamp(normalized_item["posted_at"])
            
            # 过滤长度
            text = self._extract_text(normalized_item)
            if len(text) < self.min_text_length or len(text) > self.max_text_length:
                continue
            
            normalized_data.append(normalized_item)
        
        logger.info(f"文本标准化完成 - 输出：{len(normalized_data)} 条")
        return normalized_data
    
    def _standardize_text(self, text: str) -> str:
        """标准化文本内容"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s.,!?;:()\[\]"\'-]', '', text)
        
        # 标准化货币符号
        text = re.sub(r'\$', ' USD ', text)
        text = re.sub(r'¥', ' CNY ', text)
        text = re.sub(r'€', ' EUR ', text)
        
        # 标准化百分比
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        
        return text
    
    def _normalize_timestamp(self, timestamp: Any) -> str:
        """标准化时间戳为 ISO 格式"""
        if isinstance(timestamp, str):
            # 尝试解析常见格式
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%a, %d %b %Y %H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp[:25], fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
        
        return str(timestamp)
    
    def add_sentiment_labels(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        添加情感标签
        
        Args:
            data: 原始数据列表
            
        Returns:
            添加情感标签的数据列表
        """
        logger.info(f"开始情感标注 - 数据量：{len(data)}, 模型：{self.sentiment_model}")
        
        # 加载情感分析模型
        self._load_sentiment_model()
        
        labeled_data = []
        
        for item in data:
            text = self._extract_text(item)
            if not text:
                continue
            
            # 预测情感
            sentiment_result = self._predict_sentiment(text)
            
            # 添加标签
            labeled_item = item.copy()
            labeled_item["sentiment"] = sentiment_result["label"]
            labeled_item["sentiment_score"] = sentiment_result["score"]
            
            labeled_data.append(labeled_item)
        
        logger.info(f"情感标注完成 - 输出：{len(labeled_data)} 条")
        return labeled_data
    
    def _load_sentiment_model(self):
        """加载情感分析模型"""
        if self._sentiment_pipeline is not None:
            return
        
        try:
            if self.sentiment_model == "finbert":
                from transformers import pipeline
                self._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert"
                )
                logger.info("加载 FinBERT 情感分析模型")
            
            elif self.sentiment_model == "openai":
                # 使用 OpenAI API
                if not self._check_openai_key():
                    logger.warning("OpenAI API Key 未配置，使用规则-based 情感分析")
                    self._sentiment_pipeline = "rules"
                else:
                    logger.info("使用 OpenAI API 进行情感分析")
            
            else:
                # 使用规则-based 方法
                self._sentiment_pipeline = "rules"
                logger.info("使用规则-based 情感分析")
                
        except Exception as e:
            logger.warning(f"加载情感分析模型失败：{str(e)}，使用规则-based 方法")
            self._sentiment_pipeline = "rules"
    
    def _check_openai_key(self) -> bool:
        """检查 OpenAI API Key"""
        import os
        return bool(os.getenv("OPENAI_API_KEY"))
    
    def _predict_sentiment(self, text: str) -> Dict[str, Any]:
        """预测文本情感"""
        if self._sentiment_pipeline is None:
            self._load_sentiment_model()
        
        # 使用 FinBERT
        if callable(self._sentiment_pipeline):
            try:
                # 截断过长的文本
                max_length = 512
                truncated_text = text[:max_length]
                
                result = self._sentiment_pipeline(truncated_text)[0]
                return {
                    "label": result["label"].lower(),
                    "score": float(result["score"])
                }
            except Exception as e:
                logger.warning(f"FinBERT 预测失败：{str(e)}")
        
        # 使用规则-based 方法
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """基于规则的情感分析"""
        text_lower = text.lower()
        
        positive_words = [
            "growth", "profit", "gain", "increase", "beat", "exceed", "outperform",
            "bullish", "buy", "upgrade", "positive", "strong", "record", "success",
            "上涨", "盈利", "增长", "突破", "利好", "买入", "强势"
        ]
        
        negative_words = [
            "loss", "decline", "drop", "fall", "miss", "underperform", "bearish",
            "sell", "downgrade", "negative", "weak", "fail", "risk", "warning",
            "下跌", "亏损", "下滑", "利空", "卖出", "风险", "警告"
        ]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {"label": "neutral", "score": 0.5}
        
        pos_ratio = pos_count / total
        
        if pos_ratio > 0.6:
            return {"label": "positive", "score": pos_ratio}
        elif pos_ratio < 0.4:
            return {"label": "negative", "score": 1 - pos_ratio}
        else:
            return {"label": "neutral", "score": 0.5}
    
    def extract_entities(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取命名实体
        
        Args:
            data: 原始数据列表
            
        Returns:
            添加实体信息的數據列表
        """
        if not self.entity_recognition:
            return data
        
        logger.info(f"开始实体识别 - 数据量：{len(data)}")
        
        extracted_data = []
        
        for item in data:
            text = self._extract_text(item)
            if not text:
                continue
            
            # 提取实体
            entities = self._extract_entities_from_text(text)
            
            labeled_item = item.copy()
            labeled_item["entities"] = entities
            
            extracted_data.append(labeled_item)
        
        logger.info(f"实体识别完成 - 输出：{len(extracted_data)} 条")
        return extracted_data
    
    def _extract_entities_from_text(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取实体"""
        entities = []
        
        # 提取公司名（简化：基于常见后缀）
        company_patterns = [
            r'\b([A-Z][A-Za-z]+ (?:Inc\.?|Corp\.?|Ltd\.?|LLC|Co\.?))\b',
            r'\b([A-Z][A-Za-z]+ (?:科技 | 公司 | 集团 | 股份))\b'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": "ORGANIZATION",
                    "subtype": "company"
                })
        
        # 提取人名
        person_patterns = [
            r'\b(?:CEO|CFO|President|Director) ([A-Z][a-z]+ [A-Z][a-z]+)\b',
            r'\b(?:首席执行官 | 首席财务官 | 总裁 | 董事) ([\u4e00-\u9fa5]{2,4})\b'
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": "PERSON",
                    "subtype": "executive"
                })
        
        # 提取金额
        money_patterns = [
            r'\$([\d,.]+(?: billion| million| trillion)?)',
            r'(\d+(?:\.\d+)?)\s*(?:亿 | 万)\s*(?:元 | 美元)'
        ]
        
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": "MONEY",
                    "subtype": "amount"
                })
        
        return entities
    
    def extract_keywords(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取关键词
        
        Args:
            data: 原始数据列表
            
        Returns:
            添加关键词的数据列表
        """
        if not self.keyword_extraction:
            return data
        
        logger.info(f"开始关键词提取 - 数据量：{len(data)}")
        
        extracted_data = []
        
        for item in data:
            text = self._extract_text(item)
            if not text:
                continue
            
            # 提取关键词
            keywords = self._extract_keywords_from_text(text)
            
            labeled_item = item.copy()
            labeled_item["keywords"] = keywords
            
            extracted_data.append(labeled_item)
        
        logger.info(f"关键词提取完成 - 输出：{len(extracted_data)} 条")
        return extracted_data
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 金融领域关键词
        finance_keywords = [
            "earnings", "revenue", "profit", "growth", "stock", "market",
            "investment", "dividend", "valuation", "IPO", "merger", "acquisition",
            "财报", "营收", "利润", "增长", "股票", "市场",
            "投资", "分红", "估值", "上市", "并购"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in finance_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        # 提取高频词（简化）
        words = re.findall(r'\b\w+\b', text_lower)
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in ["the", "and", "that", "with", "this", "from"]:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 添加高频词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in sorted_words[:5]:
            if word not in found_keywords:
                found_keywords.append(word)
        
        return found_keywords[:10]  # 限制最多 10 个关键词
