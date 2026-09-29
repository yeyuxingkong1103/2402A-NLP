# -*- coding: utf-8 -*-
"""集成测试统一入口：把 e2e/ 下的链路探针收成一次运行、一张总表。

覆盖（与批次 37 清单一致）：
  1. 认证落库            scripts/e2e/e2e_auth_persist.py          （自带后端起停）
  2. 长期记忆            scripts/e2e/e2e_long_term_memory.py      （服务层直连，无需后端）
  3. 审核发布            scripts/e2e/e2e_review_publish.py        （自带后端起停）
  4. PDF 解析            scripts/e2e/pdf_parse_main_path.py       （真调远程 MinerU，消耗额度）
  5. 多轮+摘要           scripts/eval/multi_turn_summary_check.py （服务层直连，无需后端）

行为约定：
  - 每项失败不阻断后续（跑完给总表）；
  - 退出码：0 = 全部通过；1 = 至少一项失败；2 = 参数/前置不满足；
  - 收尾自动跑 cleanup_e2e_probe_accounts.py 清探针账号（含 Milvus 记忆/会话/用户，
    见该脚本头部说明）；--keep-probe 跳过清理（调试用）；
  - 每项输出实时透传到控制台，同时留档 %TEMP%/integration_tests/<时间戳>/<脚本名>.log。

用法（项目根）：
    python scripts/e2e/run_integration_tests.py                 # 全部 5 项
    python scripts/e2e/run_integration_tests.py --skip-pdf      # 跳过 PDF（省 MinerU 额度）
    python scripts/e2e/run_integration_tests.py --only auth,memory
前置：Redis / MySQL / Milvus 在线 + Embedding/LLM 可用（可先跑 scripts/check_services.py）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

# 项名 → (脚本相对路径, 一句话说明)
SUITES: dict[str, tuple[str, str]] = {
    "auth": ("scripts/e2e/e2e_auth_persist.py", "认证落库（注册-重启-登录-隔离）"),
    "memory": ("scripts/e2e/e2e_long_term_memory.py", "长期记忆（写入-检索-隔离-删除-去重）"),
    "review": ("scripts/e2e/e2e_review_publish.py", "审核发布（权限-索引-留痕）"),
    "pdf": ("scripts/e2e/pdf_parse_main_path.py", "PDF 解析主路径（真调 MinerU，耗额度）"),
    "multiturn": ("scripts/eval/multi_turn_summary_check.py", "多轮+摘要（真实 LLM 压摘要）"),
}
CLEANUP = "scripts/e2e/cleanup_e2e_probe_accounts.py"


def run_suite(name: str, rel_path: str, log_dir: Path) -> tuple[bool, float, str]:
    """跑单个探针脚本；返回 (是否通过, 耗时秒, 失败摘要)。输出透传并落日志。"""
    script = PROJECT_ROOT / rel_path
    log_path = log_dir / f"{name}.log"
    print(f"\n{'=' * 64}\n[{name}] {rel_path}\n{'=' * 64}")
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            [PY, str(script)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
            tail.append(line)
            tail = tail[-20:]  # 失败时只回看最后 20 行
        code = proc.wait()
    elapsed = time.monotonic() - start
    ok = code == 0
    summary = "" if ok else "；".join(l.strip() for l in tail[-5:] if l.strip())[-300:]
    print(f"\n[{name}] 退出码={code} 耗时={elapsed:.1f}s → {'PASS' if ok else 'FAIL'}"
          + (f"（日志：{log_path}）" if ok else ""))
    return ok, elapsed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 解析（省 MinerU 额度）")
    parser.add_argument("--only", default="", help="逗号分隔项名（auth,memory,review,pdf,multiturn）")
    parser.add_argument("--keep-probe", action="store_true", help="收尾不清理探针账号（调试用）")
    args = parser.parse_args()

    selected = list(SUITES)
    if args.only:
        selected = [s for s in args.only.split(",") if s.strip() in SUITES]
        unknown = [s for s in args.only.split(",") if s.strip() not in SUITES]
        if unknown:
            print(f"未知项名：{unknown}（可用：{list(SUITES)}）")
            return 2
    if args.skip_pdf and "pdf" in selected:
        selected.remove("pdf")
    if not selected:
        print("没有可执行的项。")
        return 2

    import os
    log_dir = Path(os.environ.get("TEMP", "/tmp")) / "integration_tests" / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"集成测试开始：{len(selected)} 项；日志目录 {log_dir}")

    results: list[tuple[str, bool, float, str]] = []
    for name in selected:
        rel_path, desc = SUITES[name]
        print(f"\n▶ [{name}] {desc}")
        ok, elapsed, summary = run_suite(name, rel_path, log_dir)
        results.append((name, ok, elapsed, summary))

    # 收尾：清理探针账号（失败不阻断总表输出）
    if not args.keep_probe:
        print(f"\n{'=' * 64}\n[cleanup] 清理探针账号与从属数据\n{'=' * 64}")
        cleanup = subprocess.run([PY, str(PROJECT_ROOT / CLEANUP)], cwd=str(PROJECT_ROOT))
        cleanup_ok = cleanup.returncode == 0
    else:
        cleanup_ok = None

    # 总表
    print(f"\n{'=' * 64}\n集成测试总表\n{'=' * 64}")
    print(f"{'项':<12}{'结果':<8}{'耗时':>10}   说明")
    for name, ok, elapsed, _ in results:
        print(f"{name:<12}{'PASS' if ok else 'FAIL':<8}{elapsed:>9.1f}s   {SUITES[name][1]}")
    total_fail = sum(1 for _, ok, _, _ in results if not ok)
    print(f"\n合计 {len(results)} 项：{len(results) - total_fail} 通过 / {total_fail} 失败"
          f"；探针清理：{'完成' if cleanup_ok else ('失败（手工跑 cleanup_e2e_probe_accounts.py）' if cleanup_ok is False else '跳过')}")
    print(f"逐项日志：{log_dir}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
