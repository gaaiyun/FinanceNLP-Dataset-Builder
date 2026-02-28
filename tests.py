#!/usr/bin/env python3
"""
FinanceNLP Dataset Builder - 单元测试
"""

import unittest
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from finance_nlp_dataset import FinanceNLPBuilder
from finance_nlp_dataset.data_collector import DataCollector
from finance_nlp_dataset.data_processor import DataProcessor
from finance_nlp_dataset.dataset_builder import DatasetBuilder


class TestDataCollector(unittest.TestCase):
    """数据采集器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.collector = DataCollector(
            language="en",
            news_sources=["finnhub"],
            social_platforms=["reddit"]
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.collector.language, "en")
        self.assertIn("finnhub", self.collector.news_sources)
    
    def test_extract_text(self):
        """测试文本提取"""
        test_cases = [
            ({"content": "test content"}, "test content"),
            ({"text": "test text"}, "test text"),
            ({"title": "test title", "content": "test content"}, "test title test content"),
            ({}, ""),
        ]
        
        for item, expected in test_cases:
            result = self.collector._extract_text(item)
            self.assertEqual(result, expected)


class TestDataProcessor(unittest.TestCase):
    """数据处理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.processor = DataProcessor(
            language="en",
            dedup_threshold=0.85,
            min_text_length=10,
            max_text_length=1000,
            sentiment_model="rules"
        )
    
    def test_deduplication(self):
        """测试去重功能"""
        test_data = [
            {"id": "1", "content": "This is a test news article"},
            {"id": "2", "content": "This is a test news article"},  # 重复
            {"id": "3", "content": "Different news content"},
        ]
        
        result = self.processor.deduplicate(test_data)
        
        # 应该移除重复项
        self.assertLessEqual(len(result), len(test_data))
    
    def test_normalize_text(self):
        """测试文本标准化"""
        test_data = [
            {
                "id": "1",
                "content": "This   is   a   test",
                "published_at": "2024-01-15T10:30:00Z"
            }
        ]
        
        result = self.processor.normalize_text(test_data)
        
        self.assertEqual(len(result), 1)
        # 检查多余空格是否被移除
        self.assertNotIn("  ", result[0]["content"])
    
    def test_sentiment_analysis(self):
        """测试情感分析"""
        test_data = [
            {"id": "1", "content": "Apple stock soars to record high on strong earnings"},
            {"id": "2", "content": "Tesla shares plummet after disappointing delivery numbers"},
            {"id": "3", "content": "Market remains unchanged in quiet trading session"},
        ]
        
        result = self.processor.add_sentiment_labels(test_data)
        
        self.assertEqual(len(result), 3)
        
        # 检查情感标签
        self.assertIn("sentiment", result[0])
        self.assertIn("sentiment_score", result[0])
        
        # 第一条应该是正面
        self.assertEqual(result[0]["sentiment"], "positive")
        
        # 第二条应该是负面
        self.assertEqual(result[1]["sentiment"], "negative")
    
    def test_entity_extraction(self):
        """测试实体提取"""
        test_data = [
            {
                "id": "1",
                "content": "Apple Inc. CEO Tim Cook announced record revenue of $100 billion"
            }
        ]
        
        result = self.processor.extract_entities(test_data)
        
        self.assertEqual(len(result), 1)
        self.assertIn("entities", result[0])
    
    def test_keyword_extraction(self):
        """测试关键词提取"""
        test_data = [
            {
                "id": "1",
                "content": "Tech stocks rally as earnings season beats expectations. Apple, Microsoft lead gains."
            }
        ]
        
        result = self.processor.extract_keywords(test_data)
        
        self.assertEqual(len(result), 1)
        self.assertIn("keywords", result[0])
        self.assertGreater(len(result[0]["keywords"]), 0)


class TestDatasetBuilder(unittest.TestCase):
    """数据集构建器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = Path("./test_datasets")
        self.test_dir.mkdir(exist_ok=True)
        
        self.builder = DatasetBuilder(
            output_dir=str(self.test_dir),
            format="json",
            compress=False
        )
        
        self.test_data = [
            {
                "id": f"test_{i}",
                "type": "news",
                "title": f"Test News {i}",
                "content": f"This is test content number {i}",
                "sentiment": "positive" if i % 2 == 0 else "negative",
                "sentiment_score": 0.8 if i % 2 == 0 else 0.3
            }
            for i in range(10)
        ]
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_json_export(self):
        """测试 JSON 导出"""
        output_path = self.builder._export_json(
            self.test_data,
            "test_json"
        )
        
        self.assertTrue(os.path.exists(output_path))
        
        # 验证内容
        with open(output_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(len(loaded_data), len(self.test_data))
    
    def test_jsonl_export(self):
        """测试 JSONL 导出"""
        output_path = self.builder._export_jsonl(
            self.test_data,
            "test_jsonl"
        )
        
        self.assertTrue(os.path.exists(output_path))
        
        # 验证内容
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.assertEqual(len(lines), len(self.test_data))
    
    def test_csv_export(self):
        """测试 CSV 导出"""
        output_path = self.builder._export_csv(
            self.test_data,
            "test_csv"
        )
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_data_split(self):
        """测试数据集划分"""
        split = {"train": 0.7, "val": 0.15, "test": 0.15}
        splits = self.builder._split_data(self.test_data, split)
        
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)
        
        # 检查总数
        total = sum(len(s) for s in splits.values())
        self.assertEqual(total, len(self.test_data))
    
    def test_export_with_split(self):
        """测试带划分的数据集导出"""
        split = {"train": 0.8, "test": 0.2}
        
        output_paths = self.builder.export(
            self.test_data,
            "test_split",
            format="json",
            split=split
        )
        
        # 应该返回多个路径
        self.assertIsInstance(output_paths, list)
        
        # 检查文件是否存在
        for path in output_paths:
            self.assertTrue(os.path.exists(path))


class TestFinanceNLPBuilder(unittest.TestCase):
    """FinanceNLPBuilder 集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = Path("./test_integration")
        self.test_dir.mkdir(exist_ok=True)
        
        self.builder = FinanceNLPBuilder(
            output_dir=str(self.test_dir),
            language="en",
            sentiment_model="rules"
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.builder.collector)
        self.assertIsNotNone(self.builder.processor)
        self.assertIsNotNone(self.builder.builder)
    
    def test_clean_and_label(self):
        """测试清洗和标注流程"""
        test_data = [
            {
                "id": "1",
                "content": "Apple stock rises on strong iPhone sales"
            },
            {
                "id": "2",
                "content": "Apple stock rises on strong iPhone sales"  # 重复
            },
            {
                "id": "3",
                "content": "Tesla faces production challenges"
            }
        ]
        
        result = self.builder.clean_and_label(
            data=test_data,
            tasks=["dedup", "normalize", "sentiment"]
        )
        
        # 去重后应该少于原始数据
        self.assertLess(len(result), len(test_data))
        
        # 应该有情感标签
        for item in result:
            self.assertIn("sentiment", item)
            self.assertIn("sentiment_score", item)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestDataCollector))
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestFinanceNLPBuilder))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
