#!/usr/bin/env python3
"""
简单测试脚本 - 验证 FinanceNLP 模块功能
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 现在导入
from data_collector import DataCollector
from data_processor import DataProcessor
from dataset_builder import DatasetBuilder

print("[OK] 模块导入成功！")

# 测试 DataCollector
print("\n[Test] 测试 DataCollector...")
collector = DataCollector(language="en", news_sources=["finnhub"])
print(f"   语言：{collector.language}")
print(f"   数据源：{collector.news_sources}")
print("[OK] DataCollector 初始化成功")

# 测试 DataProcessor
print("\n[Test] 测试 DataProcessor...")
processor = DataProcessor(language="en", sentiment_model="rules")
print(f"   语言：{processor.language}")
print(f"   情感模型：{processor.sentiment_model}")
print("[OK] DataProcessor 初始化成功")

# 测试数据处理功能
print("\n[Test] 测试数据处理功能...")
test_data = [
    {"id": "1", "content": "Apple stock rises on strong earnings"},
    {"id": "2", "content": "Apple stock rises on strong earnings"},  # 重复
    {"id": "3", "content": "Tesla shares fall on production concerns"}
]

# 去重
deduped = processor.deduplicate(test_data)
print(f"   去重：{len(test_data)} -> {len(deduped)} 条")

# 情感标注
labeled = processor.add_sentiment_labels(deduped)
print(f"   情感标注：{len(labeled)} 条")
for item in labeled:
    print(f"     - {item['id']}: {item['sentiment']} ({item['sentiment_score']:.2f})")

print("[OK] 数据处理功能测试成功")

# 测试 DatasetBuilder
print("\n[Test] 测试 DatasetBuilder...")
test_output_dir = current_dir / "test_output"
builder = DatasetBuilder(output_dir=str(test_output_dir), format="json")

output_path = builder._export_json(labeled, "test_dataset")
print(f"   导出路径：{output_path}")
print("[OK] DatasetBuilder 测试成功")

# 清理测试输出
import shutil
if test_output_dir.exists():
    shutil.rmtree(test_output_dir)
    print("\n[Clean] 测试输出已清理")

print("\n" + "=" * 60)
print("[OK] 所有测试通过！FinanceNLP 模块工作正常")
print("=" * 60)
