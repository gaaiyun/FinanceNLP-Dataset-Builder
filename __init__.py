#!/usr/bin/env python3
"""
FinanceNLP Dataset Builder - 金融 NLP 数据集构建工具

主模块：提供统一的 API 接口
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

try:
    from .data_collector import DataCollector
    from .data_processor import DataProcessor
    from .dataset_builder import DatasetBuilder
except ImportError:
    # 仓库目录名含连字符（FinanceNLP-Dataset-Builder），不是合法 Python 包名，
    # 所以同时支持作为脚本（绝对 import）和作为包（相对 import）使用。
    from data_collector import DataCollector
    from data_processor import DataProcessor
    from dataset_builder import DatasetBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinanceNLPBuilder:
    """
    金融 NLP 数据集构建器
    
    整合数据采集、处理、导出全流程
    """
    
    def __init__(
        self,
        output_dir: str = "./datasets",
        language: str = "zh",
        news_sources: List[str] = None,
        social_platforms: List[str] = None,
        dedup_threshold: float = 0.85,
        min_text_length: int = 50,
        max_text_length: int = 5000,
        sentiment_model: str = "finbert",
        entity_recognition: bool = True,
        keyword_extraction: bool = True,
        export_format: str = "parquet",
        compress: bool = True
    ):
        """
        初始化构建器
        
        Args:
            output_dir: 输出目录
            language: 语言 (zh/en)
            news_sources: 新闻数据源列表
            social_platforms: 社交媒体平台列表
            dedup_threshold: 去重相似度阈值
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
            sentiment_model: 情感分析模型
            entity_recognition: 是否进行实体识别
            keyword_extraction: 是否提取关键词
            export_format: 导出格式 (json/csv/parquet)
            compress: 是否压缩输出
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.language = language
        self.export_format = export_format
        self.compress = compress
        
        # 初始化组件
        self.collector = DataCollector(
            language=language,
            news_sources=news_sources,
            social_platforms=social_platforms
        )
        
        self.processor = DataProcessor(
            language=language,
            dedup_threshold=dedup_threshold,
            min_text_length=min_text_length,
            max_text_length=max_text_length,
            sentiment_model=sentiment_model,
            entity_recognition=entity_recognition,
            keyword_extraction=keyword_extraction
        )
        
        self.builder = DatasetBuilder(
            output_dir=self.output_dir,
            format=export_format,
            compress=compress
        )
        
        logger.info(f"FinanceNLPBuilder 初始化完成 - 语言：{language}, 输出目录：{self.output_dir}")
    
    def collect_news(
        self,
        symbols: List[str],
        days: int = 7,
        sources: List[str] = None,
        categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        采集财经新闻数据
        
        Args:
            symbols: 股票代码列表
            days: 采集天数
            sources: 数据源（覆盖默认配置）
            categories: 新闻类别
            
        Returns:
            新闻数据列表
        """
        logger.info(f"开始采集新闻数据 - 股票：{symbols}, 天数：{days}")
        
        if sources is None:
            sources = self.collector.news_sources
        
        all_news = []
        
        for source in sources:
            try:
                news = self.collector.collect_financial_news(
                    symbols=symbols,
                    days=days,
                    source=source,
                    categories=categories
                )
                all_news.extend(news)
                logger.info(f"从 {source} 采集到 {len(news)} 条新闻")
            except Exception as e:
                logger.error(f"从 {source} 采集新闻失败：{str(e)}")
        
        logger.info(f"新闻采集完成 - 总计：{len(all_news)} 条")
        return all_news
    
    def parse_earnings(
        self,
        symbols: List[str],
        quarters: List[str] = None,
        years: List[int] = None,
        source: str = "sec_edgar"
    ) -> List[Dict[str, Any]]:
        """
        解析财报文本
        
        Args:
            symbols: 股票代码列表
            quarters: 季度列表 (Q1, Q2, Q3, Q4)
            years: 年份列表
            source: 数据源
            
        Returns:
            财报数据列表
        """
        logger.info(f"开始解析财报 - 股票：{symbols}")
        
        try:
            earnings = self.collector.collect_earnings_reports(
                symbols=symbols,
                quarters=quarters,
                years=years,
                source=source
            )
            logger.info(f"财报解析完成 - 总计：{len(earnings)} 份")
            return earnings
        except Exception as e:
            logger.error(f"财报解析失败：{str(e)}")
            return []
    
    def collect_social_sentiment(
        self,
        symbols: List[str],
        platforms: List[str] = None,
        days: int = 3,
        limit_per_symbol: int = 100
    ) -> List[Dict[str, Any]]:
        """
        采集社交媒体情绪数据
        
        Args:
            symbols: 股票代码/关键词列表
            platforms: 平台列表（覆盖默认配置）
            days: 采集天数
            limit_per_symbol: 每个股票的限制数量
            
        Returns:
            社交媒体数据列表
        """
        logger.info(f"开始采集社交媒体数据 - 股票：{symbols}")
        
        if platforms is None:
            platforms = self.collector.social_platforms
        
        all_social = []
        
        for platform in platforms:
            try:
                social = self.collector.collect_social_media_sentiment(
                    symbols=symbols,
                    days=days,
                    platform=platform,
                    limit=limit_per_symbol
                )
                all_social.extend(social)
                logger.info(f"从 {platform} 采集到 {len(social)} 条数据")
            except Exception as e:
                logger.error(f"从 {platform} 采集数据失败：{str(e)}")
        
        logger.info(f"社交媒体数据采集完成 - 总计：{len(all_social)} 条")
        return all_social
    
    def clean_and_label(
        self,
        data: List[Dict[str, Any]],
        tasks: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        数据清洗和标注
        
        Args:
            data: 原始数据列表
            tasks: 处理任务列表
            
        Returns:
            处理后的数据列表
        """
        if tasks is None:
            tasks = ["dedup", "normalize", "sentiment", "entities", "keywords"]
        
        logger.info(f"开始数据清洗和标注 - 任务：{tasks}, 数据量：{len(data)}")
        
        processed_data = data
        
        for task in tasks:
            try:
                if task == "dedup":
                    processed_data = self.processor.deduplicate(processed_data)
                    logger.info(f"去重完成 - 剩余：{len(processed_data)} 条")
                
                elif task == "normalize":
                    processed_data = self.processor.normalize_text(processed_data)
                    logger.info("文本标准化完成")
                
                elif task == "sentiment":
                    processed_data = self.processor.add_sentiment_labels(processed_data)
                    logger.info("情感标注完成")
                
                elif task == "entities":
                    processed_data = self.processor.extract_entities(processed_data)
                    logger.info("实体识别完成")
                
                elif task == "keywords":
                    processed_data = self.processor.extract_keywords(processed_data)
                    logger.info("关键词提取完成")
                    
            except Exception as e:
                logger.error(f"处理任务 {task} 失败：{str(e)}")
                continue
        
        logger.info(f"数据清洗和标注完成 - 最终数据量：{len(processed_data)} 条")
        return processed_data
    
    def export_dataset(
        self,
        data: List[Dict[str, Any]],
        filename: str = None,
        format: str = None,
        split: Optional[Dict[str, float]] = None
    ) -> str:
        """
        导出数据集
        
        Args:
            data: 数据列表
            filename: 输出文件名
            format: 导出格式（覆盖默认配置）
            split: 数据集划分比例 {"train": 0.8, "val": 0.1, "test": 0.1}
            
        Returns:
            输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"finance_nlp_dataset_{timestamp}"
        
        if format is None:
            format = self.export_format
        
        logger.info(f"开始导出数据集 - 文件名：{filename}, 格式：{format}")
        
        output_path = self.builder.export(
            data=data,
            filename=filename,
            format=format,
            split=split
        )
        
        logger.info(f"数据集导出完成 - 路径：{output_path}")
        return output_path
    
    def build_complete_dataset(
        self,
        symbols: List[str],
        news_days: int = 7,
        social_days: int = 3,
        quarters: List[str] = None,
        output_filename: str = None
    ) -> str:
        """
        构建完整数据集（一键式）
        
        Args:
            symbols: 股票代码列表
            news_days: 新闻采集天数
            social_days: 社交媒体采集天数
            quarters: 财报季度
            output_filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        logger.info(f"开始构建完整数据集 - 股票：{symbols}")
        
        # 1. 采集所有数据
        all_data = []
        
        # 采集新闻
        news_data = self.collect_news(symbols=symbols, days=news_days)
        all_data.extend(news_data)
        
        # 采集财报
        earnings_data = self.parse_earnings(symbols=symbols, quarters=quarters)
        all_data.extend(earnings_data)
        
        # 采集社交媒体
        social_data = self.collect_social_sentiment(symbols=symbols, days=social_days)
        all_data.extend(social_data)
        
        logger.info(f"数据采集完成 - 总计：{len(all_data)} 条")
        
        # 2. 清洗和标注
        cleaned_data = self.clean_and_label(all_data)
        
        # 3. 导出
        output_path = self.export_dataset(
            data=cleaned_data,
            filename=output_filename,
            split={"train": 0.8, "val": 0.1, "test": 0.1}
        )
        
        logger.info(f"完整数据集构建完成 - 输出：{output_path}")
        return output_path


# 便捷函数
def create_builder(**kwargs) -> FinanceNLPBuilder:
    """创建 FinanceNLPBuilder 实例的便捷函数"""
    return FinanceNLPBuilder(**kwargs)


# 版本信息
__version__ = "1.0.0"
__author__ = "gaaiyun"
__all__ = ["FinanceNLPBuilder", "create_builder"]
