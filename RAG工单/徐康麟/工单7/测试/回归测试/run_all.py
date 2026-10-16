# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 回归测试总入口。

依次执行三个用例集并输出汇总：

1. ``test_单轮问答.py``   —— 单轮问答回归（``TC-Q05/07/09/11/14/17``）
2. ``test_多轮对话.py``   —— 多轮对话回归（A 组同 session 追问 / B 组指代 / C 组 MultiTurnEngine 对照）
3. ``test_异常输入.py``   —— 异常输入回归（``TC-A01``~``TC-A07``）

工程要点
--------
* **单进程共享引擎**：三组共用同一个已预热 ``QAEngine``（多轮组再包一层 ``MultiTurnEngine``），
  只预热一次 ⇒ 对 CPU 单实例 Ollama 的占用最小（本机 Ollama 仅 1 个推理实例，必须串行、小批量）。
* **控制台全量留痕**：stdout/stderr 同时写 ``测试/测试数据/回归测试日志.txt``（tee）。
* **逐题落盘**：合并三组明细到 ``测试/测试数据/回归结果.json``，供 ``测试/生成截图.py`` 渲染真实截图。
* **退出码**：全部用例通过 ⇒ ``0``；存在失败 ⇒ ``1``；索引缺失/引擎构造失败 ⇒ ``1``（不伪造通过）。

运行：
    python 测试\\回归测试\\run_all.py                 # 全量
    python 测试\\回归测试\\run_all.py --single-limit 3  # 单轮只跑前 3 题（Ollama 忙时的小批量）
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _公共 as K  # noqa: E402

LOG_PATH = K.TEST_DATA_DIR / "回归测试日志.txt"
RESULT_PATH = K.TEST_DATA_DIR / "回归结果.json"


class Tee(io.TextIOBase):
    """把写入同时转发到原始 stdout 与内存缓冲（用于落盘全量日志）。"""

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.buffer_text = io.StringIO()

    def write(self, data: str) -> int:  # type: ignore[override]
        self.buffer_text.write(data)
        try:
            self.stream.write(data)
        except Exception:  # noqa: BLE001 —— 控制台编码异常不得影响测试
            pass
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        try:
            self.stream.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:  # type: ignore[override]
        return False


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"gd7_{name}", HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"无法加载 {name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _failed_group(group: str, exc: BaseException) -> dict[str, Any]:
    """用例集整体崩溃时，如实登记为整组失败（不静默丢弃）。"""
    case = K.new_case(f"{group}-CRASH", f"{group}用例集执行中断", group=group,
                      expect="用例集可执行完毕")
    case["异常"] = f"{type(exc).__name__}: {exc}"
    case["异常堆栈"] = traceback.format_exc()[-4000:]
    K.add_assert(case, "用例集执行完毕", False, case["异常"])
    K.finalize_case(case)
    return K.summarize(group, [case], 0.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工单7 回归测试总入口（三个用例集串行）")
    parser.add_argument("--single-limit", type=int, default=None, help="单轮组只跑前 N 题")
    parser.add_argument("--log", default=str(LOG_PATH))
    parser.add_argument("--result", default=str(RESULT_PATH))
    args = parser.parse_args(argv)

    K.bootstrap_env()
    tee = Tee(sys.stdout)
    sys.stdout = tee  # type: ignore[assignment]

    t_suite = time.perf_counter()
    facts = K.env_facts()
    ready, index_info = K.index_ready()
    print("=" * 100)
    print(f"  {K.WORK_ORDER} —— 回归测试总入口 run_all.py")
    print("=" * 100)
    print(f"开始时间   : {K.now_str()}")
    print(f"解释器     : {facts['解释器']} ({facts['解释器版本']})")
    print(f"工作目录   : {facts['工作目录']}")
    print(f"索引清单   : {index_info}")
    print(f"数据源     : {K.EVAL_DIR / '测试问题集.jsonl'}")
    print(f"串行纪律   : 单进程共享引擎，仅预热一次；不并发调用 Ollama")
    print("=" * 100)

    groups: list[dict[str, Any]] = []
    engine = None
    mt = None
    warm: dict[str, Any] = {}
    fatal = ""

    if not ready:
        fatal = "索引缺失（研发/data/index/index_manifest.json 不存在）——回归无法执行"
        print(f"\n!! {fatal}")
    else:
        print("\n[准备] 预热引擎（索引/分词/嵌入/全文倒排/生成）…")
        t0 = time.perf_counter()
        try:
            m1 = _load("test_单轮问答")
            engine, warm = m1.make_engine()
            from app.multiturn_engine import MultiTurnEngine

            mt = MultiTurnEngine(cfg=engine.cfg, engine=engine, persist_dir=None)
            # 底层引擎刚刚预热过：这里**只**构造改写器（实体表），避免二次预热抢占 Ollama。
            if hasattr(mt, "_ensure_rewriter"):
                mt._ensure_rewriter()  # noqa: SLF001 —— 有意复用已预热的底层引擎
                mt_warm = {"reused_warmup": True, "max_turns": getattr(mt, "max_turns", None)}
            else:  # pragma: no cover —— 实现变化时的兜底
                mt_warm = mt.warmup(with_generate=False)
            warm["multiturn"] = mt_warm
            print(f"       预热完成，耗时 {(time.perf_counter() - t0) * 1000:.0f} ms")
            print(f"       索引块数={warm.get('index_chunks')}｜生成后端预热="
                  f"{'ok' if (warm.get('generate') or {}).get('ok') else warm.get('generate')}")
        except Exception as exc:  # noqa: BLE001
            fatal = f"引擎构造/预热失败：{type(exc).__name__}: {exc}"
            print(f"\n!! {fatal}")
            print(traceback.format_exc()[-2000:])

    if engine is not None:
        for idx, name in enumerate(("test_单轮问答", "test_多轮对话", "test_异常输入"), start=1):
            print("\n" + "#" * 100)
            print(f"# [{idx}/3] {name}.py")
            print("#" * 100)
            group_label = {"test_单轮问答": "单轮问答", "test_多轮对话": "多轮对话",
                           "test_异常输入": "异常输入"}[name]
            t0 = time.perf_counter()
            try:
                module = _load(name)
                if name == "test_单轮问答":
                    result = module.run(engine=engine, limit=args.single_limit, warm=warm)
                elif name == "test_多轮对话":
                    result = module.run(engine=engine, mt=mt, warm=warm)
                else:
                    result = module.run(engine=engine, warm=warm)
                result["墙钟ms"] = round((time.perf_counter() - t0) * 1000, 2)
            except Exception as exc:  # noqa: BLE001 —— 单组崩溃不拖垮整套
                print(f"\n!! {name} 执行异常：{type(exc).__name__}: {exc}")
                print(traceback.format_exc()[-3000:])
                result = _failed_group(group_label, exc)
            groups.append(result)

    # ---------------------------------------------------------------- 汇总
    total_cases = sum(g["用例数"] for g in groups)
    total_pass = sum(g["通过"] for g in groups)
    total_fail = sum(g["失败"] for g in groups)
    total_ms = round((time.perf_counter() - t_suite) * 1000, 2)

    print("\n" + "=" * 100)
    print("  汇总表")
    print("=" * 100)
    rows = [[g["分组"], g["用例数"], g["通过"], g["失败"],
             f"{g['通过率'] * 100:.1f}%", f"{g['耗时ms'] / 1000:.1f}"] for g in groups]
    rows.append(["合计", total_cases, total_pass, total_fail,
                 f"{(total_pass / total_cases * 100) if total_cases else 0.0:.1f}%",
                 f"{total_ms / 1000:.1f}"])
    print(K.render_table(["用例集", "用例数", "通过", "失败", "通过率", "耗时(s)"], rows,
                         aligns=["left", "right", "right", "right", "right", "right"]))

    # 失败用例点名（不隐藏）
    failed_cases = [c for g in groups for c in g["用例"] if not c.get("通过")]
    if failed_cases:
        print("\n失败用例点名：")
        for case in failed_cases:
            first = next((a for a in case.get("断言") or [] if not a["通过"]), None)
            print(f"  - {case['用例编号']} {case['用例名称']}："
                  f"{first['名称'] if first else '无断言'} ⇒ {first['实测'] if first else ''}")
    else:
        print("\n失败用例点名：无（全部用例通过）")

    status = "PASS" if (total_fail == 0 and total_cases > 0 and not fatal) else "FAIL"
    print(f"\n结论：{status}｜用例 {total_pass}/{total_cases} 通过，失败 {total_fail}｜"
          f"总耗时 {total_ms / 1000:.1f}s")
    print(f"结束时间：{K.now_str()}")

    payload = {
        "工单": K.WORK_ORDER,
        "生成时间": K.now_str(),
        "环境": facts,
        "索引": index_info,
        "预热": warm,
        "致命错误": fatal or None,
        "分组": groups,
        "汇总": {"用例总数": total_cases, "通过": total_pass, "失败": total_fail,
                 "通过率": round(total_pass / total_cases, 4) if total_cases else 0.0,
                 "总耗时ms": total_ms, "结论": status},
        "日志": str(Path(args.log)),
    }
    K.save_json(Path(args.result), payload)
    # 同步落一份「按组」明细（与各组独立运行时 回归_<组名>.json 同名同结构，便于逐组复核）
    group_file = {"单轮问答": "回归_单轮问答.json", "多轮对话": "回归_多轮对话.json",
                  "异常输入": "回归_异常输入.json"}
    for group in groups:
        filename = group_file.get(str(group.get("分组")))
        if not filename:
            continue
        K.save_json(K.TEST_DATA_DIR / filename,
                    {"工单": K.WORK_ORDER, "生成时间": K.now_str(), "环境": facts,
                     "索引": index_info, "来源": f"{Path(args.result).name}（run_all.py 汇总）",
                     "组": group})
    print(f"逐题明细：{args.result}")
    print(f"按组明细：{'、'.join(group_file.values())}")
    print(f"控制台日志：{args.log}")

    # ---------------------------------------------------------------- 落盘日志
    sys.stdout = tee.stream  # type: ignore[assignment]
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(tee.buffer_text.getvalue(), encoding="utf-8")
    print(f"日志已落盘：{log_path}（{len(tee.buffer_text.getvalue())} 字符）")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
