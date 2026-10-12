# 工单编号：人工智能NLP-RAG-图像内容解析及检索优化
# 文件说明：RAGAS 评估。对工单 16 题跑 faithfulness / answer_relevancy / context_precision / context_recall。
# 用法：python scripts/eval_ragas.py [--limit 16] [--tag baseline]
import argparse, json, sys, warnings                                        # 参数、序列化、警告抑制
from datetime import datetime                                                # 时间戳
from pathlib import Path                                                     # 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))               # 允许 import 项目根模块
warnings.filterwarnings("ignore")                                            # RAGAS 依赖链的弃用告警刷屏，屏蔽掉
import config as C                                                           # 集中配置
from backend import rag as R                                                 # 复用问答流水线，保证评估与实际服务同一条链路


def collect(limit: int) -> list:                                             # 采集评估样本
    """逐题走真实链路：既拿到答案与上下文，也顺带记录耗时——评估与性能数据同源，避免两套口径。"""
    items = json.loads(C.QUESTIONS.read_text(encoding="utf-8"))[:limit]        # 题库
    samples = []                                                             # 评估样本
    for i, it in enumerate(items, 1):                                        # 逐题
        r = R.answer(it["question"], "auto", use_cache=False)                 # 绕开缓存，测真实链路
        contexts = [s["text"] for s in r.get("contexts", [])]                 # 检索到的上下文
        samples.append(dict(user_input=it["question"], response=r["answer"],  # RAGAS 字段名
                            retrieved_contexts=contexts or [""],              # 无上下文时占位，避免 RAGAS 报空
                            reference=it.get("answer", ""),                   # 标准答案作参考
                            _meta=dict(id=it["id"], elapsed_ms=round(r.get("elapsed", 0) * 1000),  # 附带耗时
                                       retrieval_precision=r.get("retrieval_precision"),  # 检索精确度
                                       degraded=r.get("degraded"))))          # 是否降级
        print(f"  采集 {i}/{len(items)}  id={it['id']}  {samples[-1]['_meta']['elapsed_ms']} ms  "
              f"上下文 {len(contexts)} 条", flush=True)                        # 进度
    return samples                                                           # 返回样本


def judge_llm():                                                             # 评估用裁判模型
    """用 DeepSeek 作裁判（OpenAI 兼容）；温度 0 保证打分可复现。

    坑：RAGAS 部分指标默认要 `n=3` 个生成，而 DeepSeek 只支持 `n=1`，会报
    "Invalid n value (currently only n = 1 is supported)"——实测直接把
    llm_context_precision_with_reference 打成 nan。置 bypass_n=True 后，
    包装器改发 n 条独立请求（逐条 n=1），绕开该限制。
    """
    from langchain_openai import ChatOpenAI                                  # 延迟导入
    from ragas.llms import LangchainLLMWrapper                               # RAGAS 包装器
    if not C.DEEPSEEK_API_KEY: raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法运行 RAGAS 裁判模型")  # 缺密钥明确提示
    w = LangchainLLMWrapper(ChatOpenAI(model=C.LLM_MODEL, api_key=C.DEEPSEEK_API_KEY,  # 裁判模型
                                       base_url=C.LLM_BASE_URL, temperature=0, timeout=120))  # 超时放宽，评估可慢
    w.bypass_n = True                                                        # ★不传 n，改发 n 条独立请求
    return w                                                                 # 返回包装器


def judge_emb():                                                             # 评估用嵌入模型
    """直接用本地 bge-m3，避免引入外部嵌入服务与额外费用。

    坑：RAGAS 0.4 起自家 `HuggingfaceEmbeddings` 变成了抽象基类（要求实现 aembed_documents/aembed_query），
    直接实例化会报 "Can't instantiate abstract class"。必须走 Langchain 包装器这条兼容路径。
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper                  # 兼容包装器
    from langchain_huggingface import HuggingFaceEmbeddings                  # 本地模型实现
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=C.EMBED_MODEL_PATH))  # bge-m3 本地路径


KEY = "llm_context_precision_with_reference"                                 # 本轮要盯的指标
NAN_GUARD = 0.10                                                             # 单轮 nan 比例上限，超过即整轮作废


def one_round(limit: int, rc) -> dict:                                       # 单轮完整评估（采集 + 判分）
    """跑一轮，返回指标、逐题得分，以及**该轮是否可信**。

    可信性守则是必需的，不是可选项：实测有一轮 16 题里 10 题因 DeepSeek 瞬时 402 变成 `nan`，
    聚合值算出 0.6682，而 **exit code 仍是 0**——若不检查就采信，会得出"改动让指标变差"的反向错误结论。
    故用 nan 比例作闸门，超阈值的轮次标记为作废、不进入统计。
    """
    from ragas import EvaluationDataset, evaluate                            # RAGAS 主入口
    from ragas.metrics import (AnswerRelevancy, Faithfulness,                # 四项指标
                               LLMContextPrecisionWithReference, LLMContextRecall)  # 上下文精确度与召回
    samples = collect(limit)                                                 # 采集答案与上下文
    ds = EvaluationDataset.from_list([{k: v for k, v in s.items() if not k.startswith("_")} for s in samples])  # 组装数据集
    res = evaluate(dataset=ds,                                               # 执行评估
                   metrics=[Faithfulness(), AnswerRelevancy(),               # 忠实度与答案相关性
                            LLMContextPrecisionWithReference(), LLMContextRecall()],  # 上下文精确度与召回
                   llm=judge_llm(), embeddings=judge_emb(), raise_exceptions=False,  # 单条失败不中断整体
                   run_config=rc)                                            # 低并发执行参数
    df = res.to_pandas()                                                     # 逐题结果
    metric_cols = [c for c in df.columns                                     # 指标列
                   if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    scores = {k: (round(float(v), 4) if isinstance(v, (int, float)) else None) for k, v in df[metric_cols].mean().items()}  # 各指标均分
    # 逐题导出：只报均分会掩盖"到底是哪几题在拖后腿"。按问题文本对齐，避免依赖 RAGAS 的行序。
    by_q = {str(r.get("user_input")): r for _, r in df.iterrows()}            # 问题 → 该题得分行
    per_q = []                                                               # 逐题记录
    for s in samples:                                                        # 逐个样本
        row = dict(s["_meta"])                                               # 耗时与检索精确度
        r = by_q.get(str(s["user_input"]))                                    # 对应得分行
        for c in metric_cols:                                                # 逐个指标
            v = r.get(c) if r is not None else None                           # 取值
            row[c] = round(float(v), 4) if isinstance(v, (int, float)) else None  # 数值化
        per_q.append(row)                                                    # 收集
    nan_n = sum(1 for r in per_q if r.get(KEY) is None)                       # 该指标缺失的题数
    ratio = nan_n / len(per_q) if per_q else 1.0                             # 缺失比例
    elapsed = [s["_meta"]["elapsed_ms"] for s in samples]                     # 端到端耗时
    return dict(scores=scores, n=len(samples), per_question=per_q,            # 指标与逐题
                avg_ms=round(sum(elapsed) / len(elapsed)) if elapsed else 0,  # 平均响应时间
                nan_ratio=round(ratio, 3), valid=ratio <= NAN_GUARD)          # ★可信性


def stats_of(vals: list, target: float) -> dict:                             # 多轮统计与达标概率
    """给出均值、样本标准差、95% 置信区间，以及"单次测量 ≥ 目标"的概率。

    目的是回答一个先于调参的问题：**目标本身是不是这个裁判能可靠测出来的**。
    实测同一份代码连跑两次 context_precision 为 0.7359 / 0.7793，分辨力约 ±0.05；
    在这种噪声下讨论 0.06 的差距，必须先给出区间与概率，否则只是在噪声里打转。
    """
    import math                                                              # 正态分布用 erf
    n = len(vals)                                                            # 有效轮数
    if n == 0: return dict(n=0)                                              # 无有效轮
    m = sum(vals) / n                                                        # 均值
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0  # 样本标准差
    se = sd / math.sqrt(n) if n else 0.0                                     # 均值标准误
    # 小样本用 t 分布较严谨，但 n<5 时 t 表值波动大，这里用正态近似并在报告中注明；n≥3 够看量级
    ci = (round(m - 1.96 * se, 4), round(m + 1.96 * se, 4))                   # 95% 置信区间
    p = 0.5 * (1 - math.erf((target - m) / (sd * math.sqrt(2)))) if sd > 0 else float(m >= target)  # P(单次 ≥ 目标)
    return dict(n=n, mean=round(m, 4), std=round(sd, 4), se=round(se, 4),    # 均值/标准差/标准误
                min=round(min(vals), 4), max=round(max(vals), 4),            # 极值
                ci95=ci, p_ge_target=round(p, 4))                            # 区间与达标概率


def main() -> None:                                                          # 评估主流程
    ap = argparse.ArgumentParser()                                           # 命令行
    ap.add_argument("--limit", type=int, default=16, help="评估题目数上限")        # 题量
    ap.add_argument("--tag", default="", help="结果文件名后缀，便于对照优化前后")       # 标记
    ap.add_argument("--repeat", type=int, default=1, help="重复轮数；>1 时给出均值/标准差/置信区间/达标概率")  # 复测轮数
    ap.add_argument("--target", type=float, default=0.85, help="达标目标值（算概率用）")  # 目标
    a = ap.parse_args()                                                      # 解析
    from ragas.run_config import RunConfig                                   # 执行参数
    # 坑：RAGAS 默认 max_workers=16，会 16 路并发猛打 DeepSeek 触发限流，
    # 实测 64 个作业里 18 个 TimeoutError，把 llm_context_precision_with_reference 打成 nan。
    # 降到 4 路并放宽单作业超时后恢复正常。
    rc = RunConfig(max_workers=4, timeout=300)                               # 低并发 + 宽超时
    runs = []                                                                # 各轮结果
    dump = C.ROOT / "results" / f"ragas_{datetime.now():%Y%m%d_%H%M%S}{('_' + a.tag) if a.tag else ''}.json"  # 结果文件
    dump.parent.mkdir(exist_ok=True)                                         # 确保目录存在

    def flush_out(done: bool) -> None:                                       # ★增量落盘
        """每轮跑完就写盘。

        实测教训：3 轮复测跑到第 2 轮时 API 严重拥塞（每轮 20+ 个超时、单轮要 1 小时），
        只能中断；而脚本原来只在全部跑完后才写文件，**第 1 轮的结果就此丢失**，只剩日志里一个聚合值。
        长任务必须先落盘再谈跑得完。
        """
        dump.write_text(json.dumps(dict(out, complete=done), ensure_ascii=False, indent=1), encoding="utf-8")  # 落盘

    out = {}                                                                 # 先占位，供增量写使用
    for i in range(1, a.repeat + 1):                                         # 逐轮
        print(f"\n===== 第 {i}/{a.repeat} 轮 =====", flush=True)                # 轮次提示
        r = one_round(a.limit, rc)                                            # 跑一轮
        runs.append(r)                                                        # 收集
        good_n = sum(1 for x in runs if x["valid"])                          # 目前有效轮数
        agg = {k: stats_of([x["scores"].get(k) for x in runs if x["valid"] and x["scores"].get(k) is not None], a.target)
               for k in r["scores"]}                                         # 逐指标统计（基于已有轮次）
        out.update(repeat=a.repeat, target=a.target, runs=runs, valid_runs=good_n,  # 已完成的轮次
                   dropped=len(runs) - good_n, stats=agg,                    # 作废轮数与统计
                   model=C.LLM_MODEL, embed="bge-m3")                        # 口径标注
        flush_out(done=(i == a.repeat))                                      # ★立刻落盘
        print(f"  本轮 {KEY}={r['scores'].get(KEY)}  nan 比例={r['nan_ratio']}  "
              f"{'有效' if r['valid'] else '★作废（nan 比例超阈值）'}", flush=True)  # 单轮小结
        print(f"  已落盘（累计有效 {good_n} 轮）", flush=True)                    # 落盘提示
    good = [r for r in runs if r["valid"]]                                    # 有效轮
    print("\n============ RAGAS 结果 ============")                            # 分隔
    for r, i in zip(runs, range(1, len(runs) + 1)):                           # 逐轮打印
        print(f"  第{i}轮 {r['scores'].get(KEY)}   avg {r['avg_ms']} ms   "
              f"nan {r['nan_ratio']}{'' if r['valid'] else '  ★作废'}")         # 单轮
    if a.repeat > 1:                                                         # 多轮才给统计
        print(f"\n有效轮 {len(good)}/{len(runs)}（作废 {len(runs) - len(good)} 轮）")  # 轮次
        s = agg.get(KEY) or {}                                               # 目标指标统计
        print(f"{KEY}  均值 {s.get('mean')}  标准差 {s.get('std')}  "
              f"极值 [{s.get('min')}, {s.get('max')}]")                        # 统计
        print(f"  95% 置信区间 ≈ {s.get('ci95')}   单次测量 ≥ {a.target} 的概率 = {s.get('p_ge_target')}")  # 区间与概率
        print("  ⚠ n 小时 t 分布更严谨，此处用正态近似；判读时以区间宽度为准，不要只看小数位")  # 口径提醒
        if len(good) < a.repeat:                                             # 有轮次没跑完
            print(f"  ⚠ 只完成 {len(good)}/{a.repeat} 轮，统计仅基于已完成的轮次；"
                  f"结果文件里的 complete 字段为 false")                        # 明确标注未跑满
    print("口径：裁判模型 = %s；嵌入 = bge-m3 本地；答案与上下文取自同一问答链路。" % C.LLM_MODEL)  # 口径声明
    print(f"结果已写入 {dump}")                                                # 提示


if __name__ == "__main__":                                                    # 入口
    main()                                                                    # 执行
