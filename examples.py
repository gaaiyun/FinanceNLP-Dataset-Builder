#!/usr/bin/env python3
"""
FinanceNLP Dataset Builder - 使用示例

演示如何使用 FinanceNLP 构建金融 NLP 数据集
"""

import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from finance_nlp_dataset import FinanceNLPBuilder


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)
    
    # 初始化构建器
    builder = FinanceNLPBuilder(
        output_dir="./example_datasets",
        language="en",
        news_sources=["finnhub"],  # 只使用 Finnhub
        sentiment_model="rules"  # 使用规则-based 情感分析（不需要额外模型）
    )
    
    # 采集新闻数据
    print("\n📰 采集财经新闻...")
    news_data = builder.collect_news(
        symbols=["AAPL", "TSLA"],
        days=3,
        sources=["finnhub"]
    )
    print(f"采集到 {len(news_data)} 条新闻")
    
    # 数据清洗和标注
    print("\n🧹 数据清洗和标注...")
    cleaned_data = builder.clean_and_label(
        data=news_data,
        tasks=["dedup", "normalize", "sentiment"]
    )
    print(f"处理后剩余 {len(cleaned_data)} 条数据")
    
    # 导出数据集
    print("\n📤 导出数据集...")
    output_path = builder.export_dataset(
        data=cleaned_data,
        filename="example_news_dataset",
        format="json"
    )
    print(f"数据集已导出：{output_path}")
    
    return output_path


def example_complete_pipeline():
    """完整流程示例"""
    print("\n" + "=" * 60)
    print("示例 2: 完整流程（新闻 + 财报 + 社交媒体）")
    print("=" * 60)
    
    # 初始化构建器
    builder = FinanceNLPBuilder(
        output_dir="./complete_datasets",
        language="en",
        news_sources=["finnhub", "yahoo"],
        social_platforms=["reddit"],
        sentiment_model="rules",
        entity_recognition=True,
        keyword_extraction=True
    )
    
    # 一键构建完整数据集
    print("\n🚀 启动完整数据集构建流程...")
    output_path = builder.build_complete_dataset(
        symbols=["AAPL", "NVDA"],
        news_days=5,
        social_days=3,
        quarters=["Q4"],
        output_filename="complete_finance_nlp_dataset"
    )
    
    print(f"\n✅ 完整数据集构建完成！")
    print(f"输出路径：{output_path}")
    
    return output_path


def example_custom_processing():
    """自定义处理示例"""
    print("\n" + "=" * 60)
    print("示例 3: 自定义数据处理")
    print("=" * 60)
    
    # 初始化构建器
    builder = FinanceNLPBuilder(
        output_dir="./custom_datasets",
        language="en",
        dedup_threshold=0.9,  # 更高的去重阈值
        min_text_length=100,  # 最小文本长度
        max_text_length=3000,  # 最大文本长度
        sentiment_model="rules"
    )
    
    # 采集数据
    print("\n📰 采集数据...")
    news_data = builder.collect_news(
        symbols=["MSFT", "GOOGL"],
        days=7
    )
    
    # 自定义处理流程
    print("\n🔧 自定义处理流程...")
    
    # 1. 只去重
    deduped = builder.processor.deduplicate(news_data)
    print(f"去重后：{len(deduped)} 条")
    
    # 2. 只标准化
    normalized = builder.processor.normalize_text(deduped)
    print(f"标准化后：{len(normalized)} 条")
    
    # 3. 只情感标注
    labeled = builder.processor.add_sentiment_labels(normalized)
    print(f"情感标注后：{len(labeled)} 条")
    
    # 导出
    output_path = builder.export_dataset(
        data=labeled,
        filename="custom_processed_dataset",
        format="jsonl"  # JSONL 格式
    )
    
    print(f"\n✅ 自定义处理完成！")
    print(f"输出路径：{output_path}")
    
    return output_path


def example_dataset_statistics(data_path: str):
    """查看数据集统计信息"""
    print("\n" + "=" * 60)
    print("示例 4: 数据集统计分析")
    print("=" * 60)
    
    import json
    
    # 加载数据
    print(f"\n📊 加载数据集：{data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量：{len(data)} 条\n")
    
    # 按类型统计
    type_counts = {}
    for item in data:
        item_type = item.get('type', 'unknown')
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    
    print("📋 数据类型分布:")
    for item_type, count in type_counts.items():
        print(f"  - {item_type}: {count} 条 ({count/len(data)*100:.1f}%)")
    
    # 按情感统计
    sentiment_counts = {}
    for item in data:
        sentiment = item.get('sentiment', 'unknown')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    print("\n💝 情感分布:")
    for sentiment, count in sentiment_counts.items():
        print(f"  - {sentiment}: {count} 条 ({count/len(data)*100:.1f}%)")
    
    # 按来源统计
    source_counts = {}
    for item in data:
        source = item.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print("\n📰 数据来源分布:")
    for source, count in source_counts.items():
        print(f"  - {source}: {count} 条")
    
    # 显示样本
    print("\n📝 数据样本:")
    for i, item in enumerate(data[:3], 1):
        print(f"\n样本 {i}:")
        print(f"  类型：{item.get('type', 'N/A')}")
        print(f"  标题/文本：{item.get('title', item.get('text', 'N/A'))[:100]}...")
        print(f"  情感：{item.get('sentiment', 'N/A')} ({item.get('sentiment_score', 0):.2f})")
        if 'symbols' in item:
            print(f"  股票：{', '.join(item['symbols'])}")


def main():
    """主函数"""
    print("\n" + "⭐" * 30)
    print("FinanceNLP Dataset Builder - 使用示例")
    print("⭐" * 30 + "\n")
    
    # 检查 API Keys
    if not os.getenv("FINNHUB_API_KEY"):
        print("⚠️  警告：FINNHUB_API_KEY 未配置")
        print("   请设置环境变量或使用规则-based 情感分析\n")
    
    try:
        # 运行示例
        print("请选择要运行的示例:")
        print("1. 基本使用示例")
        print("2. 完整流程示例")
        print("3. 自定义处理示例")
        print("4. 查看数据集统计（需要已有数据集）")
        print("5. 运行所有示例")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            output_path = example_basic_usage()
        elif choice == "2":
            output_path = example_complete_pipeline()
        elif choice == "3":
            output_path = example_custom_processing()
        elif choice == "4":
            data_path = input("请输入数据集路径：").strip()
            example_dataset_statistics(data_path)
            return
        elif choice == "5":
            output_path = example_basic_usage()
            output_path = example_complete_pipeline()
            output_path = example_custom_processing()
        else:
            print("无效的选项")
            return
        
        # 显示统计信息
        if output_path and os.path.exists(output_path):
            example_dataset_statistics(output_path)
        
        print("\n✅ 所有示例运行完成！")
        
    except Exception as e:
        print(f"\n❌ 运行失败：{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
