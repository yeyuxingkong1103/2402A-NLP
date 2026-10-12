# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：评估脚本。跑工单五的多轮问题（含英文与单轮变体），统计准确率与响应时间，
#           可选地补算 ragas 四项指标。
# 用法：
#   python scripts/evaluate.py                 # 只跑问答与判分，结果先落盘
#   python scripts/evaluate.py --ragas         # 在已落盘的结果上补算 ragas（可单独重跑）
#   python scripts/evaluate.py --selftest      # 用"已知错误答案"回验判分器本身是否有效
import argparse                                                                 # 命令行参数
import json                                                                     # 结果落盘
import re                                                                       # 文本归一化
import sys                                                                      # 路径
import time                                                                     # 计时
from pathlib import Path                                                        # 路径

ROOT = Path(__file__).resolve().parent.parent                                   # 项目根目录
sys.path.insert(0, str(ROOT))                                                   # 让 backend 可导入

from backend import config as C                                                 # 集中配置
from backend import pipeline, ragas_eval, redis_cache                            # 主流程、评估、会话

_NORM = re.compile(r"[\s,，、]")                                                 # 归一化时要去掉的字符


def norm(s) -> str:                                                             # 文本归一化
    """去掉空白与中英文逗号/顿号并转小写，避免 5,520 与 5520 这类格式差异被误判。"""
    return _NORM.sub("", str(s or "")).lower()                                  # 返回归一化结果


def has_all(keys, text) -> bool:                                                # 关键字段是否齐全
    """全部关键字段都出现才算命中；漏答即判错。"""
    return bool(keys) and all(norm(k) in norm(text) for k in keys)              # 逐字段判断


def has_none(keys, text) -> bool:                                               # 负样本是否都没出现
    """黑名单里一个词都不能有：用来抓"多答了无关内容"这种错误（关键词法单独用会漏判）。"""
    return not any(norm(k) in norm(text) for k in (keys or []))                 # 任一命中即失败


def is_correct(item: dict, text: str) -> bool:                                  # 答案是否正确
    """关键字段全在 **且** 负样本不在；两半都过才算对。"""
    return has_all(item.get("expect"), text) and has_none(item.get("expect_not"), text)


def ask(session_id: str, question: str, use_cache: bool = False) -> dict:       # 走一轮问答
    """调用主流程并计时；异常转成可读错误而不是让整个评估崩掉。

    默认 use_cache=False：评测要的是**真实响应时间**，命中缓存会把耗时测成几十毫秒。
    """
    t0 = time.time()                                                            # 计时
    try:                                                                        # 主流程可能失败
        r = pipeline.answer_question(session_id, question, use_cache=use_cache)  # 执行
        # 用**完整片段**而不是前端展示用的截断 snippet：截断会把 ragas 的召回率指标压低。
        return {"answer": r.get("answer", ""), "contexts": r.get("contexts", []),
                "rewritten": r.get("rewritten_query", ""), "cached": r.get("cached", False),
                "degraded": r.get("degraded", []),                              # 降级项
                "latency_ms": r.get("latency_ms", int((time.time() - t0) * 1000)),  # 端到端耗时
                "error": None}                                                  # 无错误
    except Exception as exc:                                                    # 失败
        return {"answer": "", "contexts": [], "rewritten": "", "cached": False,  # 空结果
                "degraded": ["error"], "latency_ms": int((time.time() - t0) * 1000),  # 耗时照记
                "error": f"{type(exc).__name__}: {exc}"}                        # 错误信息


def run_group(name: str, items: list, same_session: bool,
              use_cache: bool = False) -> list:                                 # 跑一组题
    """same_session=True 时在同一会话内按顺序提问（多轮指代消解必须这样跑）。"""
    sid = f"eval-{name}-{int(time.time())}"                                     # 本组会话号
    rows = []                                                                   # 逐题结果
    for i, q in enumerate(items):                                               # 逐题
        if not same_session and i > 0:                                          # 单轮组：每题独立会话
            sid = f"eval-{name}-{int(time.time())}-{i}"                         # 换会话，避免上一题实体污染
        r = ask(sid, q["question"], use_cache=use_cache)                        # 提问
        rows.append({"group": name, "id": q["id"], "question": q["question"],   # 题面
                     "ground_truth": q.get("ground_truth", ""),                 # 标准答案
                     "answer": r["answer"], "contexts": r["contexts"],          # 答案与检索片段
                     "rewritten": r["rewritten"], "cached": r["cached"],        # 改写与缓存
                     "degraded": r["degraded"], "error": r["error"],            # 降级与错误
                     "latency_ms": r["latency_ms"],                             # 耗时
                     "correct": is_correct(q, r["answer"]) if not r["error"] else False,  # 是否正确
                     "neg_pass": has_none(q.get("expect_not"), r["answer"]),    # 负样本单列，便于定位是哪半没过
                     "expect": q.get("expect", []),                             # 记录判定口径，报告里可复核
                     "expect_not": q.get("expect_not", [])})                    # 同上
    return rows                                                                 # 返回


def summarize(rows: list) -> dict:                                              # 汇总
    """统计准确率、负样本通过率与耗时分位，作为 90% / 3 秒的验收依据。"""
    if not rows:                                                                # 无样本
        return {}                                                               # 返回空
    n = len(rows)                                                               # 题数
    lat = sorted(r["latency_ms"] for r in rows)                                 # 耗时排序
    return {"total": n,                                                         # 题数
            "accuracy": round(sum(r["correct"] for r in rows) / n, 4),          # 准确率（工单要求 ≥90%）
            "neg_pass_rate": round(sum(r["neg_pass"] for r in rows) / n, 4),    # 负样本通过率
            "avg_latency_ms": int(sum(lat) / n),                                # 平均耗时
            "max_latency_ms": lat[-1],                                          # 最大耗时
            "p95_latency_ms": lat[min(n - 1, int(n * 0.95))],                   # 95 分位耗时
            "within_3s": round(sum(t <= C.EVAL_TARGET_LATENCY * 1000 for t in lat) / n, 4),  # 3 秒内完成比例
            "errors": sum(1 for r in rows if r["error"]),                       # 错误条数
            "degraded": sorted({d for r in rows for d in r["degraded"]})}       # 出现过的降级项


def selftest() -> int:                                                          # 判分器自检
    """用"已知正确/已知错误"的答案回验判分器。

    必要性：工单三曾出现"自动判分 14/14 满分、人工复核却发现混入 2 家无关企业"的情况，
    说明判分器本身必须被验证，否则准确率数字不可信。
    """
    cases = [                                                                   # (是否应判对, 期望字段, 负样本, 答案)
        (True, ["程家明"], ["赵马克"], "武汉兴图新科电子股份有限公司 法定代表人是程家明。"),      # 正确
        (False, ["程家明"], ["赵马克"], "该公司法定代表人是赵马克（Mark Zhao）。"),              # 答成了另一家
        (False, ["程家明"], ["赵马克"], "该公司法定代表人是程家明，力源信息的赵马克也是。"),        # 混入无关内容
        (True, ["6,464.51", "14,414.16"], [], "军用领域收入 6464.51 万元、14414.16 万元。"),   # 千分位差异应容忍
        (False, ["6,464.51", "14,414.16"], [], "军用领域收入 6,464.51 万元。"),                # 漏答一半
    ]
    bad = 0                                                                     # 未通过的用例数
    for should_pass, expect, expect_not, ans in cases:                          # 逐个用例
        item = {"expect": expect, "expect_not": expect_not}                     # 组装题目
        got = is_correct(item, ans)                                             # 判分
        flag = "OK " if got == should_pass else "FAIL"                          # 结果标记
        if got != should_pass:                                                  # 判分与预期不符
            bad += 1                                                            # 计数
        print(f"  [{flag}] 期望={should_pass} 实得={got} :: {ans[:40]}")           # 打印
    print(f"判分器自检：{len(cases) - bad}/{len(cases)} 通过")                       # 汇总
    return 0 if bad == 0 else 1                                                 # 有任何不符即失败退出


def main() -> int:                                                              # 主入口
    """跑全部题组并落盘；--ragas 时补算指标，--selftest 时只做判分器自检。"""
    ap = argparse.ArgumentParser(description="工单五 评估")                        # 参数解析
    ap.add_argument("--ragas", action="store_true", help="补算 ragas 四项指标")    # 补算开关
    ap.add_argument("--selftest", action="store_true", help="只做判分器自检")        # 自检开关
    ap.add_argument("--rows", type=str, default="", help="指定要补算的结果文件")     # 指定文件
    args = ap.parse_args()                                                      # 解析

    if args.selftest:                                                           # 只做自检
        return selftest()                                                       # 返回结果码

    C.EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)                        # 确保结果目录存在
    if args.ragas:                                                              # 补算模式
        # 只匹配 eval_<时间戳>.json：**不能**用 eval_*.json，否则会把 eval_warm_*.json（缓存命中组，只有 5 条）
        # 当成最新结果，静默地用一半样本算指标（实测踩过：n_samples 变成 5 而没人发现）。
        src = Path(args.rows) if args.rows else sorted(
            p for p in C.EVAL_RESULT_DIR.glob("eval_*.json")
            if not p.name.startswith("eval_warm_"))[-1]                     # 取最新的冷跑结果
        data = json.loads(src.read_text(encoding="utf-8"))                      # 读已有结果
        data["ragas"] = ragas_eval.evaluate_rows(data["rows"])                  # 补算（复用已跑出的答案与片段）
        src.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")  # 覆盖保存
        print(json.dumps(data["ragas"], ensure_ascii=False))                    # 打印指标
        return 0                                                                # 结束

    qs = json.loads(C.EVAL_FILE.read_text(encoding="utf-8"))                    # 读题集
    print("[评估] 预热模型与索引（预热耗时不计入 3 秒验收）")                          # 说明
    pipeline.load()                                                             # 装载索引
    pipeline.warmup()                                                           # 真正跑一次编码与重排，测的才是稳态耗时

    rows = []                                                                   # 全部逐题结果
    rows += run_group("multi_turn", qs["multi_turn"], same_session=True)        # 多轮组：同一会话按顺序
    rows += run_group("english", qs["english"], same_session=True)              # 英文组：同一会话
    rows += run_group("standalone", qs["standalone"], same_session=False)       # 单轮组：每题独立会话

    # 再跑一遍**开缓存**：多轮组第二遍必然命中，用来量"缓存命中时的响应时间"。
    warm: dict = {}                                                             # 缓存命中耗时
    try:                                                                        # 缓存不可用不该让评估失败
        wrows = run_group("multi_turn_warm", qs["multi_turn"], same_session=True, use_cache=True)  # 第二次同会话同问题
        warm = summarize(wrows)                                                 # 汇总
        result_warm = {"summary": warm, "rows": wrows}                          # 单独留档
        (C.EVAL_RESULT_DIR / f"eval_warm_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(  # 落盘
            json.dumps(result_warm, ensure_ascii=False, indent=1), encoding="utf-8")  # 写文件
    except Exception as exc:                                                    # 失败
        print(f"[评估] 缓存命中组跳过：{type(exc).__name__}: {exc}")               # 如实告知

    result = {"summary": summarize(rows), "groups": {}, "rows": rows,           # 汇总结果
              "warm_cache": warm}                                                   # 附带缓存命中耗时
    for g in ("multi_turn", "english", "standalone"):                           # 分组汇总
        result["groups"][g] = summarize([r for r in rows if r["group"] == g])   # 各组单独统计
    path = C.EVAL_RESULT_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"    # 结果文件（带时间戳留存）
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")  # 先落盘再考虑算指标
    print(json.dumps(result["summary"], ensure_ascii=False))                    # 打印总汇总
    for g, s in result["groups"].items():                                       # 打印各组
        print(f"  {g}: 准确率={s.get('accuracy')} 3秒内={s.get('within_3s')} 平均={s.get('avg_latency_ms')}ms")
    if warm:                                                                    # 缓存命中有数据
        print(f"  缓存命中组：平均={warm.get('avg_latency_ms')}ms 最大={warm.get('max_latency_ms')}ms")  # 打印
    print(f"结果已写入：{path}")                                                  # 提示位置
    print("如需 ragas 指标：python scripts/evaluate.py --ragas")                  # 提示下一步
    return 0                                                                    # 成功


if __name__ == "__main__":                                                      # 直接执行
    raise SystemExit(main())                                                    # 以返回码退出
