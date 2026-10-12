# 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
# 文件说明：评测层。跑工单 14 题，统计回答准确率、检索精确度、响应时间，并用 Ragas 计算四项指标。
import json, re, sys, time                                                          # 序列化、关键字段匹配、命令行参数、计时
from pathlib import Path                                                            # 结果文件路径
from . import config as C                                                           # 集中配置
from . import rag_core as P                                                         # 问答流水线


def _norm(s) -> str:                                                                # 文本归一化
    """去掉空白与中英文逗号并转小写，避免 5,520 与 5520、25.04% 与 25.04 % 这类格式差异被误判。"""
    return re.sub(r"[\s,，]", "", str(s)).lower()                                    # 返回归一化结果


def _has(expect: list, text: str) -> bool:                                          # 关键字段是否齐全
    """全部关键字段都出现才为真，用于答案判定与检索精确度判定，口径统一。"""
    return bool(expect) and all(_norm(e) in _norm(text) for e in expect)             # 逐字段判断


def _has_not(expect_not: list, text: str) -> bool:                                  # 非目标实体是否都不在
    """负样本判定：黑名单里的词一个都不能出现。工单要求"检索结果准确无误"，多答与少答同样是错。"""
    return not any(_norm(e) in _norm(text) for e in (expect_not or []))              # 任一命中即判失败


def _correct(q: dict, text: str) -> bool:                                           # 答案是否正确（修订后的口径）
    """关键字段全在 **且** 非目标实体不在。

    旧口径只查"关键字段是否都在"，对付漏答有效，但对"多答无关内容"完全失效——
    实测 id 4 混入 2 家无关企业仍被判满分，这正是 RAG 里最典型的错误
    （检索到内容相近但范围不同的表，把两张表的条目合并作答），必须由负样本拦下。
    """
    return _has(q["expect"], text) and _has_not(q.get("expect_not"), text)            # 两半都过才算对


def _pages(hits: list) -> set:                                                      # 收集命中页码
    """把每个片段的页区间展开成页码集合，用于判断是否覆盖标准答案出处页。"""
    got = set()                                                                     # 页码集合
    for h in hits: got.update(range(h["page_no"], h.get("page_end", h["page_no"]) + 1))  # 累加页区间
    return got                                                                      # 返回集合


def run(with_cache: bool = False, limit: int = 0) -> list:                           # 跑一遍工单 14 题
    """逐题走完整链路，记录答案、来源、表格定位、精确度与耗时；默认关缓存以测真实耗时。"""
    qs = json.loads(C.QUESTIONS.read_text(encoding="utf-8"))["questions"]            # 读取 14 题
    rows = []                                                                       # 逐题结果
    for q in qs[:limit or len(qs)]:                                                  # 逐题执行
        t0 = time.time()                                                            # 单题计时
        r = P.answer(q["question"], lang=q.get("lang", "auto"), use_cache=with_cache)  # 走问答链路
        hits = r.get("contexts", [])                                                 # 命中的上下文片段
        got = _pages(hits)                                                           # 命中页码集合
        rows.append(dict(id=q["id"], question=q["question"], ground_truth=q["ground_truth"],  # 题面与标准答案
                         answer=r.get("answer", ""), error=r.get("error"),              # 模型答案与错误码
                         correct=_correct(q, r.get("answer", "")),                       # 答案是否正确（关键字段全在且非目标实体不在）
                         neg_pass=_has_not(q.get("expect_not"), r.get("answer", "")),    # 负样本单独记一笔，便于定位是哪一半没过
                         ctx_hit=_has(q["expect"], " ".join(c["text"] for c in hits)),  # 检索精确度：上下文已含全部关键字段
                         page_hit=any(p in got for p in q["source_pages"]),             # 出处页命中率
                         table_hit=any(c.get("table_id") and c["page_no"] == p         # 表格定位命中：命中的是表格块且页码对得上
                                       for c in hits for p in q["source_pages"]),       # 逐片段逐出处页比对
                         locators=[c.get("locator", "") for c in hits if c.get("table_id")],  # 命中的表格定位串，写进报告备查
                         pages=sorted(got), contexts=[c["text"] for c in hits],         # 命中页码与片段全文
                         elapsed=r.get("elapsed", round(time.time() - t0, 3)),          # 端到端耗时
                         retrieval_ms=r.get("retrieval_ms"), generate_ms=r.get("generate_ms")))  # 分阶段耗时
    return rows                                                                     # 返回逐题明细


def summarize(rows: list) -> dict:                                                   # 汇总统计
    """把逐题结果汇总为准确率、检索精确度与耗时分位数，作为 90% 与 3 秒验收的依据。"""
    if not rows: return {}                                                          # 空结果直接返回
    times = sorted(r["elapsed"] for r in rows)                                        # 按耗时排序
    n = len(rows)                                                                   # 题数
    return dict(total=n, accuracy=round(sum(r["correct"] for r in rows) / n, 4),       # 回答准确率（工单 ≥90% 验收项）
                neg_pass_rate=round(sum(r["neg_pass"] for r in rows) / n, 4),          # 负样本通过率：答案中未混入非目标实体
                retrieval_precision=round(sum(r["ctx_hit"] for r in rows) / n, 4),     # 检索精确度：上下文已含答案关键字段
                page_hit_rate=round(sum(r["page_hit"] for r in rows) / n, 4),          # 出处页命中率
                table_hit_rate=round(sum(r["table_hit"] for r in rows) / n, 4),        # 表格定位命中率（本次工单的技术增量）
                avg_elapsed=round(sum(times) / n, 3), max_elapsed=round(times[-1], 3),  # 平均与最大响应时间
                p95_elapsed=round(times[min(n - 1, int(n * 0.95))], 3),                # 95 分位响应时间
                within_3s=round(sum(t <= 3 for t in times) / n, 4))                    # 3 秒内完成比例（工单 ≤3s 验收项）


def ragas_metrics(rows: list) -> dict:                                                # Ragas 四项指标
    """以 DeepSeek 为判官、bge-m3 为嵌入，计算忠实度、答案相关性、上下文召回率与精确率。"""
    try:                                                                            # ragas 需装在当前解释器，缺了要给出可操作的提示
        from openai import OpenAI                                                    # 判官大模型客户端
        from langchain_huggingface import HuggingFaceEmbeddings                      # 本地 bge-m3 嵌入
        from ragas import EvaluationDataset, SingleTurnSample, evaluate              # Ragas 入口
        from ragas.embeddings import LangchainEmbeddingsWrapper                      # 嵌入适配器
        from ragas.llms import llm_factory                                           # 判官大模型工厂
        from ragas.metrics import (AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference,  # 四项指定指标
                                   LLMContextRecall)                                 # 上下文召回率
    except ImportError as exc:                                                       # 未安装时不阻断评测主流程
        return {"error": f"当前解释器未安装 ragas（{exc}）；请用装了 ragas 的解释器执行："
                         f"python -m app.evaluate_ragas --ragas --rows <评测结果.json>"}  # 给出可执行命令
    judge = llm_factory(C.LLM_MODEL, provider="openai",                               # DeepSeek 作判官
                        client=OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL))  # 走 OpenAI 兼容接口
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(                          # 本地嵌入，不联网
        model_name=C.EMBED_MODEL_PATH, model_kwargs={"device": C.DEVICE},             # 模型路径与设备
        encode_kwargs={"normalize_embeddings": True}))                                # 归一化与向量检索口径一致
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"] or "（无答案）",  # 单条评估样本
                                reference=r["ground_truth"], retrieved_contexts=r["contexts"] or ["（无检索结果）"])  # 参考与检索上下文
               for r in rows]                                                        # 逐题构造
    scores = evaluate(EvaluationDataset(samples),                                    # 数据集
                      metrics=[Faithfulness(), AnswerRelevancy(), LLMContextRecall(),  # 四项指标
                               LLMContextPrecisionWithReference()],                   # 上下文精确率
                      llm=judge, embeddings=emb, show_progress=False, raise_exceptions=False)  # 单条失败不中断整体
    return {k: None if v != v else round(float(v), 4)                                # NaN 转 None
            for k, v in scores.to_pandas().mean(numeric_only=True).items()}           # 取各项均值


def main() -> None:                                                                  # 评测主入口
    """跑 14 题并落盘结果；--ragas 时为已有结果补算 Ragas 四项（可在另一解释器里执行）。"""
    args = sys.argv[1:]                                                              # 命令行参数
    out_dir = C.ROOT / "results"; out_dir.mkdir(exist_ok=True)                        # 结果目录
    if "--ragas" in args:                                                            # 补算 Ragas 模式（在装了 ragas 的解释器里跑）
        src = Path(args[args.index("--rows") + 1]) if "--rows" in args else sorted(out_dir.glob("eval_*.json"))[-1]  # 指定结果文件或取最新
        data = json.loads(src.read_text(encoding="utf-8"))                             # 读已有评测结果
        data["ragas"] = ragas_metrics(data["rows"])                                    # 补算四项指标（用已跑出的答案与上下文，不重复调用问答链路）
        src.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")  # 覆盖保存
        print(json.dumps(data["ragas"], ensure_ascii=False)); return                    # 打印并结束
    print("预热模型与检索链路（首次调用含模型加载，不计入 3 秒验收）")                     # 说明预热目的
    P.answer("预热", use_cache=False)                                                 # 让模型常驻，测的是稳态耗时
    rows = run(with_cache=False)                                                      # 正式跑 14 题
    summary = summarize(rows)                                                         # 汇总
    path = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"                     # 文件名带时间戳便于留档
    path.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=1), encoding="utf-8")  # 落盘
    print(json.dumps(summary, ensure_ascii=False))                                    # 打印汇总
    print(f"结果已写入：{path}")                                                       # 提示结果位置


if __name__ == "__main__":                                                           # 支持命令行直接执行
    main()                                                                           # 运行评测
