# -*- coding: utf-8 -*-
"""人工智能NLP-RAG-功能测试及评估 —— 单轮问答回归测试。

覆盖用例（编号取自 ``设计/测试用例设计.md`` 的 ``TC-*`` 清单，逐题执行其对应用例）：

======================  ==================================================
用例编号                 说明
======================  ==================================================
``TC-Q05``              事实型（招商银行：董事长/行长 + 每股现金分红）
``TC-Q07``              数值型多值（平安银行：营收/净利/增速/不良率/拨备覆盖率）
``TC-Q09``              跨文档对比（平安银行 + 招商银行 营收与增速）
``TC-Q11``              数值型（中国人寿：百万元口径 3 值 + 同比）
``TC-Q14``              数值型长数字（招商证券：元口径 + 千分位归一化）
``TC-Q17``              分析论述（中国太保：保险产品+养老社区+专业服务）
======================  ==================================================

逐题断言（**全部真实执行，非模拟**）：

1. ``未抛异常``：``QAEngine.answer`` 未向调用方抛出任何异常；
2. ``答案非空``：``r.answer`` 去空白后长度 > 0；
3. ``有引用页``：``r.cited_pages`` 非空（引用可回溯的最低要求）；
4. ``引用页合法``：每个引用页落在 ``1 ≤ page ≤ 该语料 page_count``（取自索引清单）；
5. ``耗时被记录``：``r.latency_ms`` 含 ``total`` 且 ``> 0``（同时记录端到端墙钟）。

**观测项**（真实记录、但不参与 PASS/FAIL）：关键词覆盖率、是否判对、recall@5/@10、
生成方式、是否拒答。回归测试的职责是「功能不退化」，准确率由评测官口径单独汇总，
两者不混为一谈（避免用宽松断言掩盖真实缺陷，也避免用准确率当功能判据）。

运行：
    python 测试\\回归测试\\test_单轮问答.py              # 全量 6 题
    python 测试\\回归测试\\test_单轮问答.py --limit 3    # 小批量（Ollama 独占时用）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _公共 as K  # noqa: E402

#: 覆盖用例：(题库题号, 用例编号, 用例名称)
CASES: list[tuple[str, str, str]] = [
    ("Q05", "TC-Q05", "事实型（人名+金额·招商银行）"),
    ("Q07", "TC-Q07", "数值型多值（平安银行）"),
    ("Q09", "TC-Q09", "跨文档对比（平安+招商·营收与增速）"),
    ("Q11", "TC-Q11", "数值型（中国人寿·百万元口径）"),
    ("Q14", "TC-Q14", "数值型长数字（招商证券·元口径）"),
    ("Q17", "TC-Q17", "分析论述（中国太保·新业务模式）"),
]

GROUP = "单轮问答"
TOP_K = 10


def make_engine() -> tuple[Any, dict[str, Any]]:
    """构造并预热引擎（``warmup(with_generate=True)`` 是硬前置：不预热首字可达 3.4 s）。"""
    from app.core.config import get_config
    from app.core.qa_engine import QAEngine

    engine = QAEngine(cfg=get_config())
    warm = engine.warmup(with_generate=True)
    return engine, warm


def _hit_texts(answer: Any) -> list[str]:
    """从引擎自报的命中块里取正文（用于 recall@k 观测）。"""
    texts: list[str] = []
    for item in (getattr(answer, "chunks", None) or []):
        if not isinstance(item, dict):
            continue
        for key in ("content", "text", "text_head", "quote"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
                break
    return texts


def run(engine: Any = None, *, limit: int | None = None, verbose: bool = True,
        warm: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行单轮回归；``engine`` 为 ``None`` 时自建（含预热）。返回分组结果字典。"""
    K.bootstrap_env()
    questions = K.load_question_set()
    limits = K.page_limits()

    if engine is None:
        engine, warm = make_engine()

    selected = CASES[:limit] if limit else list(CASES)
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []

    for qid, case_id, name in selected:
        row = questions.get(qid, {})
        question = str(row.get("question") or "")
        keywords = row.get("answer_keywords") or []
        case = K.new_case(case_id, name, group=GROUP, question=question,
                          expect=f"答案非空 + 引用页非空 + 引用页合法 + 耗时被记录（题号 {qid}）")
        case["题库题号"] = qid
        case["公司"] = row.get("company")
        if verbose:
            print(f"\n--- {case_id} [{qid}] {name} ---")
            print(f"问：{question[:110]}{'…' if len(question) > 110 else ''}")

        if not question:
            K.add_assert(case, "题库存在该题", False, f"题库缺少 {qid}")
            K.finalize_case(case)
            cases.append(case)
            continue

        t0 = time.perf_counter()
        answer = None
        try:
            answer = engine.answer(question, lang="auto", top_k=TOP_K,
                                   session_id=f"reg-single-{qid}", return_evidence=True,
                                   use_cache=False)
        except Exception as exc:  # noqa: BLE001 —— 异常本身就是被断言的对象
            case["异常"] = f"{type(exc).__name__}: {exc}"
            case["异常堆栈"] = traceback.format_exc()[-2000:]
            case["备注"] = "调用抛出异常（回归失败）"
        wall_ms = round((time.perf_counter() - t0) * 1000, 2)
        case["耗时ms"] = wall_ms
        case["端到端墙钟ms"] = wall_ms

        if answer is None:
            K.add_assert(case, "未抛异常", False, case["异常"])
            K.add_assert(case, "答案非空", False, "无返回对象")
            K.add_assert(case, "有引用页", False, "无返回对象")
            K.add_assert(case, "引用页合法(1..page_count)", False, "无返回对象")
            K.add_assert(case, "耗时被记录", False, "无返回对象")
            K.finalize_case(case)
            cases.append(case)
            if verbose:
                print(f"    !! {case['异常']}")
            continue

        text = str(getattr(answer, "answer", "") or "")
        cited = [int(p) for p in (getattr(answer, "cited_pages", None) or [])]
        latency = dict(getattr(answer, "latency_ms", None) or {})
        refused = bool(getattr(answer, "refused", False))
        citations = list(getattr(answer, "citations", None) or [])

        case["答案"] = text[:600]
        case["答案全文"] = text
        case["答案字数"] = len(text.strip())
        case["拒答"] = refused
        case["拒答原因"] = getattr(answer, "refusal_reason", None)
        case["引用页"] = cited
        case["引用语料"] = sorted({str(c.get("corpus") or "") for c in citations if c.get("corpus")})
        case["生成方式"] = getattr(answer, "generation_method", None)
        case["引擎耗时ms"] = latency

        # --- 断言 1/2/3/5
        K.add_assert(case, "未抛异常", True, "ok")
        K.add_assert(case, "答案非空", len(text.strip()) > 0, f"{len(text.strip())} 字")
        K.add_assert(case, "有引用页", len(cited) > 0, f"{cited}")
        total_ms = float(latency.get("total") or 0.0)
        K.add_assert(case, "耗时被记录", total_ms > 0 and wall_ms > 0,
                     f"引擎total={total_ms}ms / 墙钟={wall_ms}ms")

        # --- 断言 4：引用页必须落在该语料的真实页数范围内
        bad: list[str] = []
        for cite in citations:
            corpus = str(cite.get("corpus") or "")
            page = cite.get("page")
            limit_pages = limits.get(corpus)
            if not isinstance(page, int):
                bad.append(f"{corpus}: 页码非整数({page!r})")
            elif page < 1 or (limit_pages is not None and page > limit_pages):
                bad.append(f"{corpus}: p{page} 越界(1..{limit_pages})")
        K.add_assert(case, "引用页合法(1..page_count)", not bad,
                     "全部合法" if not bad else "; ".join(bad)[:200])

        # --- 断言 5：引用语料必须是**点名语料**
        #     依据 设计/测试用例设计.md §1 通用预期：「命中/引用语料 = 点名语料」。
        #     点名语料取题库 ``source_file``（多公司题以 ``;`` 分隔），**逐字比对完整文件名**；
        #     题库缺字段时判失败（不允许"无法核验即静默通过"）。
        expected_files = [f for f in re.split(r"[;；]", str(row.get("source_file") or "")) if f]
        cited_set = sorted({str(c.get("corpus") or "") for c in citations if c.get("corpus")})
        if not expected_files:
            K.add_assert(case, "引用语料=点名语料(逐字文件名)", False,
                         "题库缺 source_file，无法核验")
        else:
            foreign = [c for c in cited_set if c not in expected_files]
            K.add_assert(case, "引用语料=点名语料(逐字文件名)", not foreign,
                         "引用语料与点名语料一致" if not foreign
                         else f"越界引用：{[c[:46] for c in foreign]}")
        obs_files = {"点名语料": [f[:46] for f in expected_files],
                     "引用语料": [c[:46] for c in cited_set]}

        # --- 观测项（不参与 PASS/FAIL）
        obs: dict[str, Any] = {}
        try:
            cov = K.coverage_of(text, keywords)
            obs.update({"关键词命中数": cov["命中数"], "关键词总数": cov["总数"],
                        "关键词覆盖率": cov["覆盖率"], "命中关键词": cov["命中关键词"],
                        "缺失关键词": cov["缺失关键词"]})
            obs["是否判对(覆盖率>=0.6且未拒答)"] = bool(
                K.judge_module().is_correct(cov["覆盖率"], refused))
        except Exception as exc:  # noqa: BLE001
            obs["判分异常"] = f"{type(exc).__name__}: {exc}"
        try:
            texts = _hit_texts(answer)
            for k in (5, 10):
                hit, tot, rec, _h, _m = K.judge_module().retrieval_recall_at_k(texts, keywords, k)
                obs[f"recall@{k}"] = rec
        except Exception as exc:  # noqa: BLE001
            obs["召回观测异常"] = f"{type(exc).__name__}: {exc}"
        obs["SLA<=3000ms"] = bool(wall_ms <= 3000)
        obs.update(obs_files)
        case["关键实测"] = obs

        K.finalize_case(case)
        cases.append(case)

        if verbose:
            flag = "PASS" if case["通过"] else "FAIL"
            print(f"    {flag}  {case['答案字数']} 字 | 引用页 {cited} | {wall_ms:.0f}ms | "
                  f"覆盖率 {obs.get('关键词覆盖率', '-')} | 判对 {obs.get('是否判对(覆盖率>=0.6且未拒答)', '-')}"
                  f" | 生成 {case['生成方式']}")
            print(f"    答：{text[:150].replace(chr(10), ' ')}{'…' if len(text) > 150 else ''}")

    elapsed = (time.perf_counter() - started) * 1000
    result = K.summarize(GROUP, cases, elapsed)
    result["预热"] = warm
    if verbose:
        K.print_case_table(cases, f"单轮问答回归（{len(cases)} 题）")
        print(f"通过 {result['通过']} / {result['用例数']}，失败 {result['失败']}，"
              f"本组耗时 {elapsed / 1000:.1f}s")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工单7 单轮问答回归测试")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（小批量）")
    parser.add_argument("--out", default=str(K.TEST_DATA_DIR / "回归_单轮问答.json"))
    args = parser.parse_args(argv)

    facts = K.env_facts()
    ready, index_info = K.index_ready()
    print(f"== {K.WORK_ORDER} / 单轮问答回归 ==")
    print(f"解释器: {facts['解释器']} ({facts['解释器版本']})")
    print(f"索引: {index_info}")
    if not ready:
        print("!! 索引缺失，单轮回归无法执行（如实失败，不伪造结果）")

    result = run(limit=args.limit)
    payload = {"工单": K.WORK_ORDER, "生成时间": K.now_str(), "环境": facts,
               "索引": index_info, "组": result}
    K.save_json(Path(args.out), payload)
    print(f"\n逐题结果已落盘：{args.out}")
    return 0 if result["失败"] == 0 and result["用例数"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
