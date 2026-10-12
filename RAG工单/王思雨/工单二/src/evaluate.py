# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
"""评估模块：跑工单 10 个问题，统计回答准确率与响应时间，并用 RAGAS 计算四项检索质量指标。"""
import json, re, time                                                                          # 序列化、关键字段匹配、计时
from . import config as C                                                                      # 复用集中配置
from . import rag_pipeline as P                                                                # 复用问答流水线


def _norm(s) -> str:                                                                           # 文本归一化
    """去掉空白与中英文逗号并转小写，避免 5,520 与 5520 这类格式差异被误判为答错。"""
    return re.sub(r"[\s,，]", "", str(s)).lower()                                               # 返回归一化结果


def _correct(expect: list, answer: str) -> bool:                                                # 答案是否正确
    """答案同时包含该题全部关键字段才判为正确，判分口径写入测试文档备查。"""
    return all(_norm(e) in _norm(answer) for e in expect)                                        # 逐关键字段判断


def _page_hit(pages: list, contexts: list) -> bool:                                              # 检索是否命中出处页
    """检索上下文覆盖标准答案出处页即算命中，用于量化优化前后的检索精确度变化。"""
    got = set()                                                                                  # 命中的页码集合
    for c in contexts: got.update(range(c["page_no"], c.get("page_end", c["page_no"]) + 1))        # 累加每个块覆盖的页区间
    return any(p in got for p in pages)                                                           # 任一出处页命中即算命中


def run(mode: str) -> list:                                                                      # 跑一遍全部题目
    """逐题走完整链路，记录答案、命中页码、耗时与是否正确，供统计与 RAGAS 复用。"""
    rows = []                                                                                     # 逐题结果
    for q in json.loads(C.QUESTIONS.read_text(encoding="utf-8"))["questions"]:                # 遍历工单 10 道题
        t0 = time.time()                                                                           # 单题计时起点
        r = P.answer(q["question"], mode=mode, use_cache=False)                                    # 关闭缓存以测真实端到端耗时
        ctx = r.get("contexts", [])                                                                # 命中的上下文块
        rows.append(dict(id=q["id"], question=q["question"], ground_truth=q["ground_truth"],        # 题面与标准答案
                         answer=r.get("answer", ""), error=r.get("error"), contexts=[c["text"] for c in ctx],  # 模型答案与检索上下文
                         pages=[c["page_no"] for c in ctx], correct=_correct(q["expect"], r.get("answer", "")),  # 命中页码与是否答对
                         page_hit=_page_hit(q["source_pages"], ctx), elapsed=r.get("elapsed", round(time.time() - t0, 3)),  # 出处页命中与耗时
                         ctx_hit=_correct(q["expect"], " ".join(c["text"] for c in ctx)),             # 检索到的上下文是否已含答案关键字段
                         retrieval_ms=r.get("retrieval_ms"), generate_ms=r.get("generate_ms")))      # 分阶段耗时
    return rows                                                                                   # 返回逐题结果


def ragas_metrics(rows: list) -> dict:                                                             # RAGAS 四项指标
    """以 DeepSeek 为判官、bge-m3 为嵌入，计算忠实度、答案相关性、上下文精确率与召回率。"""
    from openai import OpenAI                                                                    # 延迟导入，避免影响服务启动
    from langchain_huggingface import HuggingFaceEmbeddings                                       # 本地 bge-m3 嵌入
    from ragas import evaluate, EvaluationDataset, SingleTurnSample                               # RAGAS 入口
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextRecall, LLMContextPrecisionWithReference  # 四项指定指标
    from ragas.llms import llm_factory                                                            # 判官大模型工厂
    from ragas.embeddings import LangchainEmbeddingsWrapper                                       # 嵌入适配器
    judge = llm_factory(C.LLM_MODEL, provider="openai", client=OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL))  # DeepSeek 作判官
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=C.EMBED_MODEL_PATH, model_kwargs={"device": C.DEVICE}, encode_kwargs={"normalize_embeddings": True}))  # 本地嵌入模型
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"] or "（无答案）", reference=r["ground_truth"],  # 单条评估样本
                                retrieved_contexts=r["contexts"] or ["（无检索结果）"]) for r in rows]  # 检索上下文为空时占位，避免判官报错
    scores = evaluate(EvaluationDataset(samples), metrics=[Faithfulness(), AnswerRelevancy(), LLMContextRecall(), LLMContextPrecisionWithReference()],  # 指定四项指标
                      llm=judge, embeddings=emb, show_progress=False, raise_exceptions=False)       # 单条失败不中断整体评估
    return {k: None if v != v else round(float(v), 4) for k, v in scores.to_pandas().mean(numeric_only=True).items()}  # NaN 转 None 并取均值


def summarize(rows: list) -> dict:                                                                 # 汇总统计
    """把逐题结果汇总为准确率、检索命中率与响应时间分位数，作为 90% 与 3 秒验收的依据。"""
    times = sorted(r["elapsed"] for r in rows)                                                      # 按耗时排序
    return dict(accuracy=round(sum(r["correct"] for r in rows) / len(rows), 4),                     # 回答准确率
                page_hit_rate=round(sum(r["page_hit"] for r in rows) / len(rows), 4),                # 检索命中率（按标准答案出处页）
                context_hit_rate=round(sum(r["ctx_hit"] for r in rows) / len(rows), 4),              # 检索精确度（上下文已含答案关键字段）
                avg_elapsed=round(sum(times) / len(times), 3), max_elapsed=round(times[-1], 3),      # 平均与最大响应时间
                within_3s=round(sum(t <= 3 for t in times) / len(times), 4))                         # 3 秒内完成的比例


def main() -> None:                                                                                 # 评估主入口
    """依次跑优化前与优化后，打印并落盘准确率、检索命中率、耗时与 RAGAS 四项指标。"""
    out = {}                                                                                        # 两种模式的结果
    print("预热模型与检索链路（首次调用含模型加载，不计入 3 秒验收）")                                    # 说明预热目的
    P.answer("预热", mode="optimized", use_cache=False)                                              # 先跑一次让模型常驻，测的是稳态耗时
    for mode in ("baseline", "optimized"):                                                           # 先基线后优化，顺序固定便于对比
        print(f"\n===== 评测模式：{mode} =====")                                                     # 进度提示
        rows = run(mode)                                                                             # 跑全部题目
        out[mode] = dict(summary=summarize(rows), ragas=ragas_metrics(rows), rows=rows)               # 汇总 + RAGAS + 逐题明细
        print(json.dumps(out[mode]["summary"], ensure_ascii=False))                                   # 打印汇总指标
        print(json.dumps(out[mode]["ragas"], ensure_ascii=False))                                     # 打印 RAGAS 指标
    path = C.ROOT / "results" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"                              # 文件名带时间戳便于留档
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")                  # 落盘完整结果
    print(f"\n结果已写入：{path}")                                                                    # 提示结果位置


if __name__ == "__main__":                                                                           # 支持命令行直接执行
    main()                                                                                           # 运行评估
