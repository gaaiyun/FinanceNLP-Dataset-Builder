#!/usr/bin/env python3
"""
数据集构建模块

支持多种输出格式：
- JSON/JSONL
- CSV
- Parquet
- HuggingFace Dataset
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """数据集构建器"""
    
    def __init__(
        self,
        output_dir: str = "./datasets",
        format: str = "parquet",
        compress: bool = True
    ):
        """
        初始化数据集构建器
        
        Args:
            output_dir: 输出目录
            format: 默认导出格式
            compress: 是否压缩
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.format = format
        self.compress = compress
        
        logger.info(f"DatasetBuilder 初始化完成 - 输出目录：{self.output_dir}")
    
    def export(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        format: str = None,
        split: Optional[Dict[str, float]] = None
    ) -> str:
        """
        导出数据集
        
        Args:
            data: 数据列表
            filename: 文件名
            format: 导出格式
            split: 数据集划分
            
        Returns:
            输出文件路径
        """
        if format is None:
            format = self.format
        
        logger.info(f"开始导出数据集 - 格式：{format}, 数据量：{len(data)}")
        
        if split:
            # 划分数据集
            splits = self._split_data(data, split)
            output_paths = []
            
            for split_name, split_data in splits.items():
                split_filename = f"{filename}_{split_name}"
                output_path = self._export_single(
                    data=split_data,
                    filename=split_filename,
                    format=format
                )
                output_paths.append(output_path)
            
            # 保存划分信息
            self._save_split_info(filename, split, output_paths)
            
            return output_paths[0] if len(output_paths) == 1 else output_paths
        else:
            return self._export_single(data, filename, format)
    
    def _split_data(
        self,
        data: List[Dict[str, Any]],
        split: Dict[str, float]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        划分数据集
        
        Args:
            data: 完整数据集
            split: 划分比例 {"train": 0.8, "val": 0.1, "test": 0.1}
            
        Returns:
            划分后的数据集
        """
        import random
        
        # 随机打乱
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        splits = {}
        
        start_idx = 0
        for split_name, ratio in split.items():
            if split_name == "test":
                # 最后一部分包含所有剩余数据
                end_idx = total
            else:
                end_idx = start_idx + int(total * ratio)
            
            splits[split_name] = shuffled[start_idx:end_idx]
            start_idx = end_idx
        
        # 记录划分信息
        for split_name, split_data in splits.items():
            logger.info(f"数据集划分 - {split_name}: {len(split_data)} 条 ({len(split_data)/total*100:.1f}%)")
        
        return splits
    
    def _export_single(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        format: str
    ) -> str:
        """导出单个数据集文件"""
        
        if format == "json":
            return self._export_json(data, filename)
        elif format == "jsonl":
            return self._export_jsonl(data, filename)
        elif format == "csv":
            return self._export_csv(data, filename)
        elif format == "parquet":
            return self._export_parquet(data, filename)
        elif format == "hf_dataset":
            return self._export_huggingface(data, filename)
        else:
            logger.warning(f"不支持的格式：{format}，使用 JSON 格式")
            return self._export_json(data, filename)
    
    def _export_json(self, data: List[Dict[str, Any]], filename: str) -> str:
        """导出为 JSON 格式"""
        output_path = self.output_dir / f"{filename}.json"
        
        if self.compress:
            output_path = self.output_dir / f"{filename}.json.gz"
            import gzip
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON 数据集已导出 - 路径：{output_path}, 大小：{len(data)} 条")
        return str(output_path)
    
    def _export_jsonl(self, data: List[Dict[str, Any]], filename: str) -> str:
        """导出为 JSONL 格式（每行一个 JSON 对象）"""
        output_path = self.output_dir / f"{filename}.jsonl"
        
        if self.compress:
            output_path = self.output_dir / f"{filename}.jsonl.gz"
            import gzip
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"JSONL 数据集已导出 - 路径：{output_path}, 大小：{len(data)} 条")
        return str(output_path)
    
    def _export_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """导出为 CSV 格式"""
        output_path = self.output_dir / f"{filename}.csv"
        
        if not data:
            logger.warning("数据为空，跳过 CSV 导出")
            return str(output_path)
        
        # 获取所有字段
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        # 展平嵌套字段
        flattened_data = []
        for item in data:
            flat_item = {}
            for key, value in item.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        flat_item[f"{key}_{k}"] = v
                elif isinstance(value, list):
                    flat_item[key] = json.dumps(value, ensure_ascii=False)
                else:
                    flat_item[key] = value
            flattened_data.append(flat_item)
        
        # 写入 CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = sorted(all_fields)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_data)
        
        logger.info(f"CSV 数据集已导出 - 路径：{output_path}, 大小：{len(data)} 条")
        return str(output_path)
    
    def _export_parquet(self, data: List[Dict[str, Any]], filename: str) -> str:
        """导出为 Parquet 格式"""
        try:
            import pandas as pd
            
            output_path = self.output_dir / f"{filename}.parquet"
            
            # 转换为 DataFrame
            df = pd.DataFrame(data)
            
            # 处理嵌套结构
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
            
            # 导出为 Parquet
            df.to_parquet(output_path, index=False, engine='pyarrow')
            
            logger.info(f"Parquet 数据集已导出 - 路径：{output_path}, 大小：{len(data)} 条")
            return str(output_path)
            
        except ImportError:
            logger.warning("pandas 或 pyarrow 未安装，回退到 JSON 格式")
            return self._export_json(data, filename)
        except Exception as e:
            logger.error(f"Parquet 导出失败：{str(e)}，回退到 JSON 格式")
            return self._export_json(data, filename)
    
    def _export_huggingface(self, data: List[Dict[str, Any]], filename: str) -> str:
        """导出为 HuggingFace Dataset 格式"""
        try:
            from datasets import Dataset
            
            output_dir = self.output_dir / filename
            
            # 创建 Dataset
            dataset = Dataset.from_list(data)
            
            # 保存
            dataset.save_to_disk(str(output_dir))
            
            logger.info(f"HuggingFace Dataset 已导出 - 路径：{output_dir}, 大小：{len(data)} 条")
            return str(output_dir)
            
        except ImportError:
            logger.warning("datasets 库未安装，回退到 JSON 格式")
            return self._export_json(data, filename)
        except Exception as e:
            logger.error(f"HuggingFace 导出失败：{str(e)}，回退到 JSON 格式")
            return self._export_json(data, filename)
    
    def _save_split_info(
        self,
        filename: str,
        split: Dict[str, float],
        output_paths: List[str]
    ):
        """保存数据集划分信息"""
        info_path = self.output_dir / f"{filename}_info.json"
        
        info = {
            "filename": filename,
            "split_ratio": split,
            "output_paths": output_paths,
            "created_at": datetime.now().isoformat(),
            "format": self.format
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集划分信息已保存 - 路径：{info_path}")
    
    def create_dataset_card(self, filename: str, metadata: Dict[str, Any]) -> str:
        """
        创建数据集说明卡片
        
        Args:
            filename: 数据集文件名
            metadata: 元数据
            
        Returns:
            说明卡片路径
        """
        card_path = self.output_dir / f"{filename}_README.md"
        
        content = f"""# {metadata.get('name', filename)} 数据集

## 概述

{metadata.get('description', '金融 NLP 数据集')}

## 基本信息

- **创建时间**: {datetime.now().strftime('%Y-%m-%d')}
- **语言**: {metadata.get('language', 'zh/en')}
- **数据量**: {metadata.get('size', '未知')} 条
- **格式**: {metadata.get('format', 'JSON/Parquet')}

## 数据来源

{metadata.get('sources', '- 财经新闻\n- 财报文档\n- 社交媒体')}

## 数据字段

{self._generate_field_description()}

## 使用示例

```python
import json
import pandas as pd

# 加载 JSON 格式
with open('{filename}.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 加载 Parquet 格式
df = pd.read_parquet('{filename}.parquet')
```

## 许可证

{metadata.get('license', 'MIT')}

## 引用

如果您在研究中使用了本数据集，请引用：

```bibtex
@dataset{{finance_nlp_{datetime.now().year},
  title = {{{metadata.get('name', filename)}}},
  year = {{{datetime.now().year}}},
  author = {{{metadata.get('author', 'OpenClaw FinanceNLP Team')}}}
}}
```
"""
        
        with open(card_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"数据集说明卡片已创建 - 路径：{card_path}")
        return str(card_path)
    
    def _generate_field_description(self) -> str:
        """生成字段描述"""
        return """
### 通用字段

- `id`: 唯一标识符
- `type`: 数据类型 (news/earnings/social_media)
- `language`: 语言

### 新闻数据字段

- `title`: 新闻标题
- `content`: 新闻内容
- `source`: 新闻来源
- `symbols`: 相关股票代码
- `published_at`: 发布时间
- `sentiment`: 情感标签 (positive/negative/neutral)
- `sentiment_score`: 情感得分
- `entities`: 命名实体
- `keywords`: 关键词

### 财报数据字段

- `symbol`: 股票代码
- `quarter`: 季度
- `year`: 年份
- `filing_date`: 提交日期
- `document_text`: 文档全文
- `sections`: 章节内容
- `metrics`: 财务指标

### 社交媒体数据字段

- `platform`: 平台 (twitter/reddit)
- `text`: 文本内容
- `author`: 作者
- `posted_at`: 发布时间
- `engagement`: 互动数据 (likes/comments/shares)
- `sentiment`: 情感标签
"""
