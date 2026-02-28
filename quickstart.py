#!/usr/bin/env python3
"""
FinanceNLP Dataset Builder - 快速开始脚本

5 分钟快速体验金融 NLP 数据集构建！
"""

import os
import sys
from pathlib import Path

# 检查依赖
def check_dependencies():
    """检查并安装依赖"""
    print("🔧 检查依赖包...")
    
    required_packages = ["requests", "pandas"]
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"⚠️  缺少依赖包：{', '.join(missing)}")
        print("📦 正在安装依赖...")
        os.system("pip install -r requirements.txt")
    else:
        print("✅ 依赖包检查通过")


def quick_start():
    """快速开始流程"""
    print("\n" + "⭐" * 30)
    print("FinanceNLP Dataset Builder - 快速开始")
    print("⭐" * 30 + "\n")
    
    # 检查依赖
    check_dependencies()
    
    # 检查 API Keys
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    
    if not finnhub_key:
        print("\n⚠️  警告：FINNHUB_API_KEY 未配置")
        print("   可以使用演示模式（使用示例数据）")
        use_demo = input("   是否使用演示模式？(y/n): ").strip().lower()
        
        if use_demo != 'y':
            print("\n💡 提示：请配置环境变量后重新运行")
            print("   复制 .env.example 为 .env 并填入 API Keys")
            return
    else:
        print("\n✅ API Keys 配置正确")
    
    # 导入模块
    print("\n📦 加载模块...")
    try:
        from finance_nlp_dataset import FinanceNLPBuilder
    except Exception as e:
        print(f"❌ 模块加载失败：{str(e)}")
        return
    
    # 初始化构建器
    print("\n🚀 初始化构建器...")
    builder = FinanceNLPBuilder(
        output_dir="./quickstart_datasets",
        language="en",
        sentiment_model="rules",  # 使用规则-based，不需要额外模型
        entity_recognition=True,
        keyword_extraction=True
    )
    
    # 选择功能
    print("\n请选择要执行的操作:")
    print("1. 采集财经新闻并构建数据集")
    print("2. 使用演示数据（不需要 API Key）")
    print("3. 查看示例数据集统计")
    
    choice = input("\n请输入选项 (1-3): ").strip()
    
    if choice == "1":
        # 采集真实数据
        if not finnhub_key:
            print("\n❌ 需要先配置 FINNHUB_API_KEY")
            return
        
        symbols = input("请输入股票代码（用逗号分隔，如 AAPL,TSLA）: ").strip()
        if not symbols:
            symbols = "AAPL,TSLA"
        
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        print(f"\n📰 开始采集 {len(symbol_list)} 只股票的数据...")
        
        # 采集新闻
        news_data = builder.collect_news(
            symbols=symbol_list,
            days=3,
            sources=["finnhub"]
        )
        print(f"✅ 采集到 {len(news_data)} 条新闻")
        
        # 处理数据
        print("\n🧹 处理数据...")
        cleaned_data = builder.clean_and_label(
            data=news_data,
            tasks=["dedup", "normalize", "sentiment", "keywords"]
        )
        print(f"✅ 处理完成，共 {len(cleaned_data)} 条数据")
        
        # 导出数据
        print("\n📤 导出数据集...")
        output_path = builder.export_dataset(
            data=cleaned_data,
            filename="quickstart_news_dataset",
            format="json"
        )
        print(f"✅ 数据集已导出：{output_path}")
        
        # 显示统计
        show_statistics(cleaned_data)
        
    elif choice == "2":
        # 演示模式
        print("\n🎭 使用演示数据...")
        
        demo_data = [
            {
                "id": "demo_1",
                "type": "news",
                "title": "Apple Reports Record Q4 Earnings",
                "content": "Apple Inc. announced record-breaking revenue driven by strong iPhone sales and services growth. The company exceeded analyst expectations.",
                "source": "demo",
                "symbols": ["AAPL"],
                "language": "en"
            },
            {
                "id": "demo_2",
                "type": "news",
                "title": "Tesla Stock Surges on Production Milestone",
                "content": "Tesla shares jumped 8% after the company announced it reached a new production record at its Shanghai factory.",
                "source": "demo",
                "symbols": ["TSLA"],
                "language": "en"
            },
            {
                "id": "demo_3",
                "type": "news",
                "title": "Microsoft Azure Growth Accelerates",
                "content": "Microsoft reported accelerating Azure cloud revenue growth, beating Wall Street estimates for the quarter.",
                "source": "demo",
                "symbols": ["MSFT"],
                "language": "en"
            },
            {
                "id": "demo_4",
                "type": "social_media",
                "platform": "twitter",
                "symbol": "AAPL",
                "text": "$AAPL looking strong! Best earnings ever! 🚀",
                "author": "tech_investor",
                "language": "en"
            },
            {
                "id": "demo_5",
                "type": "social_media",
                "platform": "reddit",
                "symbol": "TSLA",
                "text": "Tesla production numbers are impressive. Bullish on $TSLA",
                "author": "wsb_trader",
                "language": "en"
            }
        ]
        
        print(f"✅ 加载 {len(demo_data)} 条演示数据")
        
        # 处理数据
        print("\n🧹 处理数据...")
        cleaned_data = builder.clean_and_label(
            data=demo_data,
            tasks=["normalize", "sentiment", "keywords"]
        )
        print(f"✅ 处理完成")
        
        # 导出数据
        print("\n📤 导出数据集...")
        output_path = builder.export_dataset(
            data=cleaned_data,
            filename="demo_dataset",
            format="json"
        )
        print(f"✅ 数据集已导出：{output_path}")
        
        # 显示统计
        show_statistics(cleaned_data)
        
    elif choice == "3":
        # 查看统计
        print("\n📊 查看示例数据集统计...")
        
        # 查找已有数据集
        dataset_dir = Path("./quickstart_datasets")
        if not dataset_dir.exists():
            print("❌ 未找到数据集目录")
            return
        
        json_files = list(dataset_dir.glob("*.json"))
        if not json_files:
            print("❌ 未找到 JSON 格式的数据集")
            return
        
        print(f"\n找到 {len(json_files)} 个数据集文件:")
        for i, f in enumerate(json_files, 1):
            print(f"  {i}. {f.name}")
        
        file_idx = input("\n请选择文件编号：").strip()
        try:
            file_path = json_files[int(file_idx) - 1]
        except (IndexError, ValueError):
            print("❌ 无效的选择")
            return
        
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        show_statistics(data)
    
    else:
        print("❌ 无效的选项")
        return
    
    print("\n" + "⭐" * 30)
    print("快速开始完成！")
    print("⭐" * 30)
    print("\n💡 提示:")
    print("  - 查看 examples.py 了解更多使用示例")
    print("  - 查看 README.md 了解完整文档")
    print("  - 运行 tests.py 运行单元测试")
    print()


def show_statistics(data):
    """显示数据统计"""
    import json
    
    if not data:
        print("\n📊 数据为空")
        return
    
    print("\n" + "=" * 60)
    print("📊 数据集统计")
    print("=" * 60)
    
    print(f"\n总数据量：{len(data)} 条\n")
    
    # 按类型统计
    type_counts = {}
    for item in data:
        item_type = item.get('type', 'unknown')
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    
    print("📋 数据类型分布:")
    for item_type, count in type_counts.items():
        pct = count / len(data) * 100
        print(f"  - {item_type}: {count} 条 ({pct:.1f}%)")
    
    # 按情感统计
    sentiment_counts = {}
    for item in data:
        sentiment = item.get('sentiment', 'unknown')
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    if sentiment_counts:
        print("\n💝 情感分布:")
        for sentiment, count in sentiment_counts.items():
            pct = count / len(data) * 100
            print(f"  - {sentiment}: {count} 条 ({pct:.1f}%)")
    
    # 显示样本
    print("\n📝 数据样本:")
    for i, item in enumerate(data[:3], 1):
        print(f"\n样本 {i}:")
        title = item.get('title', item.get('text', 'N/A'))
        print(f"  类型：{item.get('type', 'N/A')}")
        print(f"  标题/文本：{title[:80]}...")
        if 'sentiment' in item:
            print(f"  情感：{item['sentiment']} ({item.get('sentiment_score', 0):.2f})")
        if 'keywords' in item:
            print(f"  关键词：{', '.join(item['keywords'][:5])}")


if __name__ == "__main__":
    try:
        quick_start()
    except KeyboardInterrupt:
        print("\n\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 错误：{str(e)}")
        import traceback
        traceback.print_exc()
