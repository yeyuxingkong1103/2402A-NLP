#!/usr/bin/env python3
"""从**原始留痕**自动生成详细的微调报告（用户要求："监控、记录数据，给我一份详细报告"）。

为什么自动生成：手写报告容易漏项、也容易事后找理由。这里四类留痕缺一不可，缺了就在报告里
显式写「缺失（未记录）」，**不猜不留白**；判定用**预注册标准**（`docs/FINETUNE-REPORT.md` §六）。

留痕与来源：
* 数据：`<data-dir>/stats.json` + `train.jsonl`（长度统计）
* 训练：`<run-dir>/train_report.json` + `train_log.jsonl`（逐步 loss/显存/吞吐）
* 评测：`--baseline` 与 `--lora-eval` 两份 **逐题 JSONL**（`scripts/eval_answers.py` 的产物）

用法：
    python scripts/report_finetune.py --tag v4-lora \
        --run-dir /root/autodl-tmp/lora/legal-v1 --data-dir data/sft \
        --baseline eval/results/v5-base-27b.jsonl --lora-eval eval/results/v4-lora.jsonl \
        --out eval/results/FINETUNE-REPORT-v4-lora.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legal_rag.sft_report import (  # noqa: E402
    arm_quality,
    arm_table, compare_eval, render_report, render_suite_report, summarize_curve,
    token_stats,
)


def write_text_lf(path: Path, text: str) -> None:
    """写文本文件，**强制 LF 行尾**。

    为什么不能用 `Path.write_text`：Windows 上它会把 ``\\n`` 翻成 ``\\r\\n``，而仓库约定是 LF
    （部署脚本还有专门的 LF 闸门）。报告是在 Windows 本地生成的，必须显式指定 newline。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [警告] 读不了 {path}（{type(exc).__name__}: {exc}）")
        return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _scores(path: Path) -> dict[str, dict]:
    """从逐题结果里抽出 ``{id: score}``（score 就是判分字典）。"""
    scores: dict[str, dict] = {}
    for record in _load_jsonl(path):
        item_id = str(record.get("id") or "")
        score = record.get("score") or {}
        if item_id and score:
            scores[item_id] = score
    return scores


def _parse_pair(raw: str, kind: str, errors: list[str]) -> tuple[str, str] | None:
    """解析 ``TAG=PATH``；格式不对就记一条错误（不静默丢掉）。"""
    if "=" not in raw:
        errors.append(f"--{kind} 需要 TAG=PATH 形式，收到 {raw!r}")
        return None
    tag, _, path = raw.partition("=")
    tag, path = tag.strip(), path.strip()
    if not tag or not path:
        errors.append(f"--{kind} 的 TAG/PATH 不能为空：{raw!r}")
        return None
    return tag, path


def _per_step_tokens(report_json: dict | None, points: list[dict]) -> float | None:
    """每个**优化步**的 token 数 = ``batch × grad_accum × 平均长度``，从留痕反推。

    怎么反推：``batch × grad_accum ≈ train_samples × epochs ÷ steps``（steps 取日志最后一步），
    再乘 ``avg_tokens_per_sample``。训练侧本来就知道这个常数，但报告是**离线**读留痕的，
    所以只能这样还原——好在它只依赖可信字段，且对"新/旧/少乘 stride"三种日志口径都适用。
    """
    if not report_json or not points:
        return None
    steps = max((point.get("step") or 0) for point in points) or 0
    per_sample = report_json.get("avg_tokens_per_sample")
    samples = report_json.get("train_samples")
    epochs = report_json.get("epochs")
    if not steps or not all(isinstance(x, (int, float)) and x > 0
                            for x in (per_sample, samples, epochs)):
        return None
    batch_accum = max(1, round(samples * epochs / steps))
    return float(batch_accum) * float(per_sample)


def parse_inference(raw: str) -> dict | None:
    """解析 ``名称=显存GB``；名字里可以自带等号（例："LoRA 训练峰值（r=32）=31.65"）。

    从**最后一个** ``=`` 切分——用 ``partition`` 会被名字里的等号截断，
    报告里就会出现 ``| LoRA 训练峰值（四轮最大，r | 32）=31.65 |`` 这种坏行（真踩过）。
    """
    name, _, gb = raw.rpartition("=")
    if not name.strip() or not gb.strip():
        return None
    try:
        float(gb.strip())
    except ValueError:
        return None
    return {"name": name.strip(), "vram_gb": gb.strip()}


def _suite(args) -> int:
    """**汇总模式**：一次生成多个训练 run + 多个评测臂的报告。"""
    errors: list[str] = []
    run_specs = [pair for raw in args.run if (pair := _parse_pair(raw, "run", errors))]
    arm_specs = [pair for raw in args.arm if (pair := _parse_pair(raw, "arm", errors))]

    runs: dict[str, tuple[dict | None, dict | None]] = {}
    for tag, raw_path in run_specs:
        run_dir = Path(raw_path)
        report_json = _load_json(run_dir / "train_report.json")
        points = _load_jsonl(run_dir / "train_log.jsonl")
        # 每步 token = batch×grad_accum×平均长度；由 train_report 的样本数/epoch/步数反推
        # batch×grad_accum。给了它，summarize_curve 能把吞吐**精确**折算成真值——
        # 这一步是必需的：v5d 的日志用中间版口径记的（rate 少算 10 倍），只有这个式子救得回来。
        per_step_tokens = _per_step_tokens(report_json, points)
        curve = summarize_curve(points, per_step_tokens=per_step_tokens) if points else None
        if curve and not curve.get("points"):
            curve = None
        if report_json is None and curve is None:
            errors.append(f"run {tag} 既没有 train_report.json 也没有 train_log.jsonl：{run_dir}")
        runs[tag] = (report_json, curve)

    arms: dict[str, dict[str, dict]] = {}
    quality: dict[str, dict] = {}
    for tag, raw_path in arm_specs:
        path = Path(raw_path)
        scores = _scores(path)
        if not scores:
            errors.append(f"臂 {tag} 的逐题结果读不到（{raw_path}）")
        arms[tag] = scores
        records = _load_jsonl(path)
        if records:
            quality[tag] = arm_quality(records)

    inference: list[dict] = []
    for raw in args.inference:
        parsed = parse_inference(raw)
        if parsed is None:
            errors.append(f"--inference 需要 名称=显存GB 形式（且 GB 是数字），收到 {raw!r}")
            continue
        inference.append(parsed)

    data_dir = Path(args.data_dir)
    data_stats_raw = _load_json(data_dir / "stats.json")
    train_records = _load_jsonl(data_dir / "train.jsonl")
    data_stats = token_stats(train_records) if train_records else None
    reproduce = None
    if args.reproduce_file:
        raw_reproduce = _load_json(Path(args.reproduce_file))
        if isinstance(raw_reproduce, dict):
            reproduce = raw_reproduce.get("commands")
        elif isinstance(raw_reproduce, list):
            reproduce = raw_reproduce
        if not reproduce:
            errors.append(f"复现命令文件没给出命令：{args.reproduce_file}")

    audit = None
    fix_sweep = None
    if args.token_audit_json:
        raw_audit = _load_json(Path(args.token_audit_json))
        if not isinstance(raw_audit, dict):
            errors.append(f"截断审计读不到：{args.token_audit_json}")
        elif "legacy_right_truncate_max_len_2048" in raw_audit:
            # 合并后的证据文件：旧实现（复现事故）逐文件读数 + 修复后的窗口扫描
            audit = (raw_audit.get("legacy_right_truncate_max_len_2048") or {}).get("files")
            fix_sweep = raw_audit.get("fixed_sweep")
            if not audit:
                errors.append(f"审计文件里没有旧实现读数：{args.token_audit_json}")
        else:
            audit = raw_audit

    data_files = None
    if args.data_files_json:
        raw_files = _load_json(Path(args.data_files_json))
        if isinstance(raw_files, dict) and isinstance(raw_files.get("files"), dict):
            data_files = [{"name": name, **value}
                          for name, value in raw_files["files"].items() if value.get("present")]
        if not data_files:
            errors.append(f"实测数据文件表读不到：{args.data_files_json}")

    report = render_suite_report(
        tag=args.tag, runs=runs, arms=arms, baseline=args.baseline_arm,
        noise_items=args.noise_items, env=_load_json(Path(args.env_json)) if args.env_json else None,
        headline=args.headline,
        data=data_stats_raw, data_stats=data_stats, data_files=data_files,
        data_notes=args.data_note, audit=audit, fix_sweep=fix_sweep,
        eval_notes=args.eval_note,
        quality=quality, inference=inference, reproduce=reproduce)
    if errors:
        report += "\n## 附：生成时的告警（未静默丢弃）\n\n" + "\n".join(f"- {e}" for e in errors) + "\n"

    target = Path(args.out) if args.out else Path(f"eval/results/FINETUNE-SUITE-{args.tag}.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(target, report)

    print("=== 汇总报告要点 ===")
    print(f"  训练 run：{len(runs)} 个 -> {', '.join(runs) or '无'}")
    for tag, (report_json, curve) in runs.items():
        vram = (report_json or {}).get("peak_vram_gb")
        loss = (report_json or {}).get("train_loss")
        print(f"    {tag}: loss={loss} 峰值显存={vram}GB "
              f"曲线点={len((curve or {}).get('vram_points') or [])}")
    print(f"  评测臂：{len(arms)} 个 -> {', '.join(arms) or '无'}")
    if arms:
        table = arm_table(arms, baseline=args.baseline_arm, noise_items=args.noise_items)
        print(f"  题数={table['n']} 抖动地板={args.noise_items:g} 题（基线 {args.baseline_arm or '未指定'}）")
        for row in table["rows"]:
            delta = row["delta_items_vs_baseline"]
            delta_text = "-" if delta is None else f"{delta:+.1f}"
            print(f"    {row['tag']}: 通过率={row['ok_rate']} 相对基线={delta_text} 题")
    for line in errors:
        print(f"  [警告] {line}")
    print(f"\n已写：{target}")
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="生成详细微调报告")
    parser.add_argument("--tag", default="v4-lora")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--data-dir", default="data/sft")
    parser.add_argument("--baseline", default="", help="基线逐题 JSONL")
    parser.add_argument("--lora-eval", default="", help="微调后逐题 JSONL")
    parser.add_argument("--out", default="")
    # —— 汇总模式（可同时给多个训练 run 与多个评测臂）——
    parser.add_argument("--run", action="append", default=[],
                        help="TAG=训练目录（可多次），目录里应有 train_report.json 与 train_log.jsonl")
    parser.add_argument("--arm", action="append", default=[],
                        help="TAG=逐题结果JSONL（可多次），每个臂一份")
    parser.add_argument("--baseline-arm", default="", help="汇总表里作为基线的臂名（相对差值以它为准）")
    parser.add_argument("--noise-items", type=float, default=4.0,
                        help="同条件重复测量的抖动地板（题数），差值小于它的判为不可辨")
    parser.add_argument("--env-json", default="", help="环境事实 JSON（字典），原样写进报告第零节")
    parser.add_argument("--inference", action="append", default=[],
                        help="名称=显存GB（可多次），推理侧实测显存")
    parser.add_argument("--reproduce-file", default="", help="复现命令 JSON（含 commands 列表）")
    parser.add_argument("--data-files-json", default="",
                        help="实测数据文件表 JSON（含 files 字典：行数/正负/提示含证据）")
    parser.add_argument("--data-note", action="append", default=[],
                        help="数据留痕的冲突/说明（可多次），原样写进报告")
    parser.add_argument("--token-audit-json", default="",
                        help="真 tokenizer 的截断审计 JSON（训练信号有没有被切掉）")
    parser.add_argument("--eval-note", action="append", default=[],
                        help="评测留痕的说明（可多次），写在评测表下面")
    parser.add_argument("--headline", action="append", default=[],
                        help="结论要点（可多次），写在报告最前面")
    args = parser.parse_args()

    if args.run or args.arm:
        return _suite(args)

    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir) if args.run_dir else None

    data_stats_raw = _load_json(data_dir / "stats.json")
    train_records = _load_jsonl(data_dir / "train.jsonl")
    data_stats = token_stats(train_records) if train_records else None

    run = _load_json((run_dir / "train_report.json") if run_dir else Path("__missing__"))
    curve = summarize_curve(_load_jsonl(run_dir / "train_log.jsonl")) if run_dir else None
    if curve and not curve.get("points"):
        curve = None

    comparison = None
    if args.baseline and args.lora_eval:
        baseline_scores = _scores(Path(args.baseline))
        lora_scores = _scores(Path(args.lora_eval))
        if baseline_scores and lora_scores:
            comparison = compare_eval(baseline_scores, lora_scores)
        else:
            print(f"  [警告] 评测结果读不到：baseline={len(baseline_scores)} 题 "
                  f"lora={len(lora_scores)} 题（报告会把这一段标为缺失）")

    report = render_report(tag=args.tag, data=data_stats_raw, run=run, curve=curve,
                           comparison=comparison, data_stats=data_stats)
    target = Path(args.out) if args.out else Path(f"eval/results/FINETUNE-REPORT-{args.tag}.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(target, report)

    print("=== 报告要点 ===")
    if data_stats_raw:
        print(f"  数据：{data_stats_raw.get('train')} 训练 / {data_stats_raw.get('val')} 验证；"
              f"接受 {data_stats_raw.get('accepted')} / 拒绝 {data_stats_raw.get('rejected')}")
    else:
        print("  数据：缺失（未记录 stats.json）")
    if run:
        print(f"  训练：loss={run.get('train_loss')} 耗时={run.get('elapsed_seconds')}s "
              f"峰值显存={run.get('peak_vram_gb')}GB")
    else:
        print("  训练：缺失（未记录 train_report.json）")
    print(f"  曲线：{curve if curve else '缺失（未记录 train_log.jsonl）'}")
    if comparison:
        print(f"  评测：可比 {comparison['questions']} 题；转正 {len(comparison['turned_green'])} / "
              f"转负 {len(comparison['turned_red'])}；P8 转正 {comparison['p8_turned_green']}")
        print(f"  指标差值：{comparison['deltas']}")
    else:
        print("  评测：缺失（需要 baseline 与 lora 两份逐题结果）")
    print(f"\n已写：{target}")

    if not any((data_stats_raw, run, comparison)):
        print("!! 三类留痕都没有：先跑数据生成/训练/评测，再来生成报告", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
