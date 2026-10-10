# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
"""一键评估出报告。

用法：
  python scripts/run_eval.py                # full_04 中文
  python scripts/run_eval.py full_04 en     # full_04 英文（按提问语言作答）

英文模式说明：标准要点（answer_key）仅中文，逐字覆盖率无意义，故只统计
检索指标与耗时，answer_accuracy 记为 null。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag04.config import get_settings, ensure_dirs        # noqa: E402
from rag04.obs.logging import setup_logging                # noqa: E402
from rag04.eval.runner import run_eval, write_report       # noqa: E402


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full_04"
    lang = sys.argv[2] if len(sys.argv) > 2 else "zh"

    s = get_settings(mode)
    ensure_dirs(s)
    log = setup_logging(s, f"rag04.eval.{mode}")

    result = run_eval(s, use_english=(lang == "en"))

    suffix = "" if lang == "zh" else "_en"
    md = write_report(result, Path(s.reports_dir) / f"检索精确度报告_{mode}{suffix}.md")
    js = Path(s.reports_dir) / f"eval_{mode}{suffix}.json"
    js.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("报告已生成：%s", md)
    log.info("指标：%s", result["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
