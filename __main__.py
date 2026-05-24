"""FinanceNLP-Dataset-Builder CLI（v2）。

子命令：
    collect     从免 key 数据源拉新闻（yfinance / rss / 合成）
    label       用 LLM 给文本按自定义 schema 打标签
    pipeline    一条龙：collect → label → 存 JSONL
    list-presets  列内置标签 schema
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_collect(args) -> int:
    from scripts.free_sources import (
        fetch_rss, fetch_yfinance_news, synthetic_samples,
    )
    if args.source == "yfinance":
        if not args.ticker:
            sys.stderr.write("[error] yfinance 源需要 --ticker\n")
            return 1
        try:
            items = fetch_yfinance_news(args.ticker, max_items=args.max)
        except ImportError as e:
            sys.stderr.write(f"[error] {e}\n")
            return 2
    elif args.source == "rss":
        if not args.url:
            sys.stderr.write("[error] rss 源需要 --url\n")
            return 1
        try:
            items = fetch_rss(args.url, max_items=args.max)
        except Exception as e:
            sys.stderr.write(f"[error] {e}\n")
            return 3
    else:  # synthetic
        items = synthetic_samples(n=args.max, seed=args.seed)

    sys.stderr.write(f"[ok] 拉到 {len(items)} 条样本\n")
    payload = [s.to_dict() for s in items]
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in payload),
            encoding="utf-8")
        sys.stderr.write(f"[ok] 写入 JSONL {args.output}\n")
    else:
        for d in payload[:5]:
            print(json.dumps(d, ensure_ascii=False, indent=2))
        if len(payload) > 5:
            print(f"... 还有 {len(payload) - 5} 条")
    return 0


def cmd_label(args) -> int:
    from scripts.llm_labeler import (
        LLMClient, LLMLabeler, LLMNotAvailable, LabelScheme, PRESETS,
    )
    # 读输入
    if args.input:
        lines = Path(args.input).read_text(encoding="utf-8").splitlines()
        # 支持 JSONL（取 text 或 title 字段）
        texts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                texts.append(obj.get("text") or obj.get("title") or "")
            except json.JSONDecodeError:
                texts.append(line)  # 纯文本一行一条
        texts = [t for t in texts if t]
    else:
        sys.stderr.write("[error] 需要 --input <file>\n")
        return 1

    # 选 schema
    if args.preset in PRESETS:
        scheme = PRESETS[args.preset]
    elif args.schema_json:
        scheme = LabelScheme(fields=json.loads(args.schema_json),
                             description=args.schema_desc or "")
    else:
        sys.stderr.write(f"[error] 用 --preset {list(PRESETS)} 或 --schema-json\n")
        return 1

    client = LLMClient(backend=args.backend)
    if not client.is_available():
        sys.stderr.write(
            f"[error] {args.backend} 缺 API key。设环境变量 "
            f"{args.backend.upper()}_API_KEY 后重试。\n"
        )
        return 2

    labeler = LLMLabeler(llm_client=client)
    sys.stderr.write(f"[info] 用 {client.backend} ({client.model}) 标 {len(texts)} 条...\n")
    samples = labeler.label_batch(texts, scheme, skip_errors=True)

    payload = [s.to_dict() for s in samples]
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in payload),
            encoding="utf-8")
        sys.stderr.write(f"[ok] 写入 {args.output}\n")
    else:
        for d in payload[:5]:
            print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0


def cmd_pipeline(args) -> int:
    """collect → label 一条龙。"""
    import tempfile
    # collect 到临时 JSONL
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".jsonl", mode="w", encoding="utf-8")
    tmp.close()
    coll_args = argparse.Namespace(
        source=args.source, ticker=args.ticker, url=args.url,
        max=args.max, seed=args.seed, output=tmp.name,
    )
    code = cmd_collect(coll_args)
    if code != 0:
        return code

    # label 到最终输出
    label_args = argparse.Namespace(
        input=tmp.name, preset=args.preset,
        schema_json=args.schema_json, schema_desc=args.schema_desc,
        backend=args.backend, output=args.output,
    )
    return cmd_label(label_args)


def cmd_list_presets(args) -> int:
    from scripts.llm_labeler import PRESETS
    print(f"{'name':<18} fields")
    print("-" * 70)
    for name, scheme in PRESETS.items():
        print(f"{name:<18} {list(scheme.fields)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="finnlp",
        description="金融 NLP 数据集构建：免 key 数据源 + LLM 自定义标签"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("collect", help="拉免 key 数据源")
    sp.add_argument("--source", default="synthetic",
                    choices=["yfinance", "rss", "synthetic"])
    sp.add_argument("--ticker")
    sp.add_argument("--url", help="RSS feed URL")
    sp.add_argument("--max", type=int, default=10)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("-o", "--output", help="输出 JSONL 路径")
    sp.set_defaults(func=cmd_collect)

    sp = sub.add_parser("label", help="LLM 按 schema 标注")
    sp.add_argument("--input", required=True, help="JSONL（含 text/title）或纯文本一行一条")
    sp.add_argument("--preset", choices=["sentiment_3way", "event_type"])
    sp.add_argument("--schema-json", help="自定义 schema JSON")
    sp.add_argument("--schema-desc", default="")
    sp.add_argument("--backend", default="deepseek",
                    choices=["openai", "anthropic", "deepseek"])
    sp.add_argument("-o", "--output", help="输出 JSONL")
    sp.set_defaults(func=cmd_label)

    sp = sub.add_parser("pipeline", help="collect → label 一条龙")
    sp.add_argument("--source", default="synthetic",
                    choices=["yfinance", "rss", "synthetic"])
    sp.add_argument("--ticker")
    sp.add_argument("--url")
    sp.add_argument("--max", type=int, default=10)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--preset", default="sentiment_3way",
                    choices=["sentiment_3way", "event_type"])
    sp.add_argument("--schema-json")
    sp.add_argument("--schema-desc", default="")
    sp.add_argument("--backend", default="deepseek",
                    choices=["openai", "anthropic", "deepseek"])
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(func=cmd_pipeline)

    sp = sub.add_parser("list-presets", help="列内置标签 schema")
    sp.set_defaults(func=cmd_list_presets)

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
