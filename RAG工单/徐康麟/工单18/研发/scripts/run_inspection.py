# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

脚本：研发/scripts/run_inspection.py
职责：真实数据集质检驱动脚本（任务四）。
  * 默认输入 = IMDR 数据集 documents/（1700 个 PDF）
  * 支持 --sample N 抽样口径（先抽样跑通、再全量）
  * 支持 --bench N 单份耗时基准（用于决定"抽样 or 全量"）
  * 报告落到 研发/data/reports/，并额外落一份带时间戳的副本到 测试/测试数据/

用法::

    python 研发/scripts/run_inspection.py --bench 20
    python 研发/scripts/run_inspection.py --sample 200
    python 研发/scripts/run_inspection.py --folder <dir> --workers 4      # 全量
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document_quality_assessment import DocumentQualityAssessmentSkill, load_config  # noqa: E402
from document_quality_assessment.config import REPO_ROOT  # noqa: E402
from document_quality_assessment.skill import collect_files, process_one  # noqa: E402

DEFAULT_DATASET = Path(r"E:\gao6gongdan\_共享输入\original_problems\documents")


def bench(folder: Path, n: int, cfg) -> dict:
    """单份耗时基准：串行处理前 N 份，给出 avg/p50/p95，用于选择抽样或全量口径。"""
    entries = collect_files(folder, None, cfg)[:n]
    times: list[float] = []
    t0 = time.perf_counter()
    for i, e in enumerate(entries, 1):
        rec = process_one(e, cfg)
        times.append(float(rec.get("elapsed_sec", 0)))
        if i % 10 == 0:
            print(f"  基准进度 {i}/{len(entries)} …", flush=True)
    total = time.perf_counter() - t0
    times.sort()
    def q(p: float) -> float:
        if not times:
            return 0.0
        idx = min(len(times) - 1, int(round((len(times) - 1) * p)))
        return times[idx]
    return {
        "files": len(entries),
        "wall_clock_sec": round(total, 3),
        "avg_sec_per_file": round(sum(times) / len(times), 4) if times else 0.0,
        "p50_sec": round(q(0.5), 4),
        "p95_sec": round(q(0.95), 4),
        "max_sec": round(max(times), 4) if times else 0.0,
        "extrapolated_full_sec_serial": round(sum(times) / len(times) * 1700, 1) if times else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="工单18 文档质量评估 Skill 驱动脚本")
    ap.add_argument("--folder", default=str(DEFAULT_DATASET), help="待质检目录")
    ap.add_argument("--bench", type=int, default=0, help="跑 N 份单份耗时基准后退出")
    ap.add_argument("--sample", type=int, default=0, help="抽样 N 份（0=全量）")
    ap.add_argument("--limit", type=int, default=None, help="等价于 --sample（沿用配置 sample_limit 时不用给）")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--tag", default="", help="产物文件名后缀标记，例如 full / sample200")
    ap.add_argument("--copy-to", default="测试/测试数据", help="额外复制一份报告到该目录（相对仓库根）")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    folder = Path(args.folder)
    if not folder.exists():
        print(f"[错误] 目录不存在：{folder}", file=sys.stderr)
        return 2

    if args.bench:
        print(f"== 单份耗时基准：{args.bench} 份（串行）==")
        result = bench(folder, args.bench, cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        out = REPO_ROOT / "优化" / "评估结果" / "单份耗时基准.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出：{out}")
        return 0

    limit = args.sample or args.limit
    tag = args.tag or (f"sample{limit}" if limit else "full")
    json_out = REPO_ROOT / "研发" / "data" / "reports" / f"quality_report_{tag}.json"
    html_out = REPO_ROOT / "研发" / "data" / "reports" / f"quality_report_{tag}.html"

    skill = DocumentQualityAssessmentSkill(cfg)
    result = skill.run(
        folder=folder,
        resume=args.resume,
        limit=limit,
        workers=args.workers,
        json_path=json_out,
        html_path=html_out,
        extra_meta={"dataset": str(folder), "tag": tag, "limit": limit or 0},
    )
    summary = result.to_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 复制到 测试/测试数据/（工单要求结果落该目录）
    if args.copy_to:
        dest = REPO_ROOT / args.copy_to
        dest.mkdir(parents=True, exist_ok=True)
        for src, name in ((json_out, f"质检报告_{tag}.json"), (html_out, f"质检报告_{tag}.html")):
            if src.exists():
                shutil.copy2(src, dest / name)
                print(f"已复制：{dest / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
