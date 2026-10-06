# -*- coding: utf-8 -*-
"""只读子进程运行器：在独立进程里调用**工单1 权威判分口径** ``Evaluator.check_answer``。

工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化

为什么必须用子进程（实测结论，不是推测）：
    * 工单3 的产品包名也是 ``app``（``app.core.*``），与工单1 同名；
      在同一进程里先 ``import app``（工单3）后再 ``from app.core.evaluator import …``
      会命中 ``sys.modules['app']`` 缓存 → ``ModuleNotFoundError``，只能降级到本地副本，
      **拿不到权威口径**；
    * 子进程把 ``工单1/研发`` 放在 ``sys.path[0]``，``app`` 唯一指向工单1，导入必然成功。

只读保证（红线）：
    * ``-B`` + ``sys.dont_write_bytecode = True`` → 不在工单1 下生成任何 ``.pyc``；
    * 用 ``Evaluator.__new__(Evaluator)`` 绕过 ``__init__``（``__init__`` 会 ``mkdir`` 结果目录、
      连 SQLite）——``check_answer`` 只用静态方法，无需实例状态；
    * 本运行器**只读**输入对（answer/golden），**只写**调用方指定的输出文件（位于工单3 内）。

用法（由 ``研发/scripts/evaluate.py`` 调用，一般不手工执行）：
    python -B 优化/脚本/ref_judge_runner.py <pairs.json> <out.json> <工单1/研发 绝对路径>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORK_ORDER = "人工智能NLP-RAG-PDF文档的表格解析及检索优化"


def emit(event: str, **fields: object) -> None:
    """打印一行结构化 JSON 事件（stdout 由调用方重定向到文件留痕）。"""
    payload = {"event": event, "module": "ref_judge_runner", "work_order": WORK_ORDER,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    """入口：读入对 → 逐条权威判分 → 写结果（退出码 0 成功；2 入参错；1 运行错）。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3:
        emit("runner.bad_args", level="ERROR", argv=args)
        return 2
    pairs_path, out_path, ref_dev = (Path(args[0]), Path(args[1]), str(args[2]))
    started = time.perf_counter()
    emit("func.enter", func="ref_judge_runner.main",
         inputs={"pairs": str(pairs_path), "out": str(out_path), "ref_dev": ref_dev})

    sys.dont_write_bytecode = True
    sys.path.insert(0, ref_dev)
    try:
        from app.core.evaluator import Evaluator  # noqa: PLC0415 —— 只读动态导入权威口径
        import app.core.evaluator as evaluator_module  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 —— 显式失败：由调用方决定是否降级
        emit("func.error", level="ERROR", func="ref_judge_runner.import", error_type=type(exc).__name__,
             message=str(exc))
        return 1

    evaluator = Evaluator.__new__(Evaluator)                  # 绕过 __init__（不建目录、不连库）
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        emit("func.error", level="ERROR", func="ref_judge_runner.main", error_type="ValueError",
             message="pairs.json 顶层必须是数组")
        return 2

    rows: list[dict[str, object]] = []
    for pair in pairs:
        qid = pair.get("id")
        t0 = time.perf_counter()
        try:
            ok, reason = evaluator.check_answer(str(pair.get("answer") or ""), str(pair.get("golden") or ""))
        except Exception as exc:  # noqa: BLE001 —— 单条失败不影响其它题，但必须在结果里显式标注
            emit("runner.judge_failed", level="ERROR", qid=qid, error_type=type(exc).__name__, message=str(exc))
            rows.append({"id": qid, "ok": None, "reason": f"{type(exc).__name__}: {exc}", "error": True})
            continue
        rows.append({"id": qid, "ok": bool(ok), "reason": str(reason),
                     "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3)})

    payload = {
        "work_order": WORK_ORDER,
        "judge": "工单1 研发/app/core/evaluator.py::Evaluator.check_answer（只读子进程、绕过 __init__）",
        "fuzzy_threshold": float(getattr(evaluator_module, "FUZZY_THRESHOLD", 0.62)),
        "evaluator_file": str(getattr(evaluator_module, "__file__", "")),
        "count": len(rows),
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("func.exit", func="ref_judge_runner.main", rows=len(rows),
         elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
         outputs={"ok_true": sum(1 for row in rows if row.get("ok") is True), "out": str(out_path)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 —— 顶层兜底：非零退出，绝不静默
        emit("runner.failed", level="ERROR", error_type=type(exc).__name__,
             message=str(exc), stack=__import__("traceback").format_exc())
        raise SystemExit(1)
