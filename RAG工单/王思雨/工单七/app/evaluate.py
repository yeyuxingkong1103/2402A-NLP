# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估 —— 评估层：10 题跑通链路 + RAGAS 四指标
# 三条纪律：① 作答先落盘再算指标（长任务中途失败不至于全丢）；② 四指标逐指标串行（混跑把并发翻 4 倍触发限流，样本被打成 nan）；
#           ③ nan 一律记 None 并单独上报有效样本数，绝不把 nan 当数字。
import json, time, statistics                                       # 落盘 / 计时 / 均值
from datetime import datetime                                       # 结果文件名时间戳
from . import config as C                                           # 配置
from .rag_core import answer                                        # 复用同一条问答链路，保证"评估的就是上线的"
def collect(limit=None, mode="hybrid"):                             # 第一段：逐题作答，产出 RAGAS 原始输入
    qs = json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))   # 读题集文件
    qs = qs["questions"] if isinstance(qs, dict) else qs            # 兼容纯列表与 {"questions":[...]} 两种写法
    qs = qs[:limit] if limit else qs                                # 可限量先小样本试跑
    rows = []                                                       # 结果行
    for i, q in enumerate(qs, 1):                                   # 逐题
        t0 = time.time()                                            # 计时
        try:                                                        # 单题失败不应中断整轮
            o = answer(q["question"], mode=mode, use_cache=False)    # 关缓存：评估要测真实链路，不能被缓存掩盖
            meta = [{"company": c["company"], "year": c["year"], "page": c["page"], "src": c["src"], "rscore": c["rscore"]} for c in o["contexts"]]  # 片段出处信息
            row = dict(q, answer=o["answer"], latency_ms=o["latency_ms"], contexts=[c["text"] for c in o["contexts"]], ctx_meta=meta, retrieval_stat=o["retrieval_stat"], error="")  # RAGAS 只吃纯文本上下文
        except Exception as e:                                      # 任何异常都写进行里，不吞掉
            row = dict(q, answer="", latency_ms=round((time.time() - t0) * 1000), contexts=[], ctx_meta=[], retrieval_stat={}, error=f"{type(e).__name__}: {e}")  # 如实记录
        rows.append(row)                                            # 收行
        print(f"[eval] {i}/{len(qs)} {q['question'][:24]}… {row['latency_ms']}ms {row['error'][:50]}")  # 进度可见
    C.RESULTS.mkdir(exist_ok=True)                                  # 确保结果目录存在
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")                  # 批次号
    (C.RESULTS / f"raw_{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")  # 先落盘
    return rows, tag                                                # 返回结果与批次号
def _judge():                                                       # 构造 RAGAS 的裁判大模型（同样走 DeepSeek）
    from openai import OpenAI                                       # 延迟导入；裁判客户端与生成端分开，避免并发互相拖累
    from ragas.llms import llm_factory                              # ragas 的 LLM 工厂（导入很慢，故延迟）
    cli = OpenAI(api_key=C.LLM_API_KEY, base_url=C.LLM_BASE_URL, timeout=180, max_retries=3)  # 裁判给足超时与重试
    kw = dict(reasoning_effort="none") if (C.LLM_REASONING_EFFORT or "flash" in C.LLM_MODEL.lower()) else {}  # 推理型/flash 模型必须显式关思考
    return llm_factory(C.LLM_MODEL, provider="openai", client=cli, max_tokens=4096, **kw)  # max_tokens 必须给足：判官一次要输出多句判定，截断会抛 IncompleteOutputException 把样本打成 nan
def _emb():                                                         # 构造 RAGAS 的嵌入模型（答案相关性指标要用）
    from langchain_huggingface import HuggingFaceEmbeddings; from ragas.embeddings import LangchainEmbeddingsWrapper  # 本地 bge-m3 → ragas 接口
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=C.EMBED_MODEL_PATH, model_kwargs={"device": C.DEVICE}))  # 指定设备
def ragas_metrics(rows):                                            # 第二段：算 RAGAS 四指标
    from ragas import EvaluationDataset, SingleTurnSample, evaluate   # 0.4.x 新版 API
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall  # 检索与生成各两项
    from ragas.run_config import RunConfig                           # 并发与超时控制
    ds = EvaluationDataset(samples=[SingleTurnSample(user_input=r["question"], response=r["answer"], reference=r["ground_truth"], retrieved_contexts=r["contexts"]) for r in rows])  # reference 即标准答案
    plan = (("faithfulness", Faithfulness()), ("answer_relevancy", AnswerRelevancy(strictness=1)), ("context_recall", LLMContextRecall()), ("context_precision", LLMContextPrecisionWithReference()))  # 逐一串行，绝不并行四指标
    out, valid = {}, {}                                             # 指标均值 / 各指标有效样本数
    for name, m in plan:                                            # 逐指标
        t0 = time.time()                                            # 计时
        try:                                                        # 单指标失败不影响其它指标
            rep = evaluate(ds, metrics=[m], llm=_judge(), embeddings=_emb(), run_config=RunConfig(max_workers=C.EVAL_MAX_WORKERS, timeout=180, max_retries=3), raise_exceptions=False)  # 不抛异常，失败样本落成 nan
            vals = rep.to_pandas()[m.name].tolist()                  # 用指标自身名字取列（如 llm_context_precision_with_reference），不能写死简称
        except Exception as e:                                      # 整体失败（如 402 余额不足）
            vals = []; print(f"[ragas] {name} 失败：{type(e).__name__}: {e}")  # 如实打印原因
        clean = [float(v) for v in vals if v == v]                   # nan != nan，用它滤掉无效值
        out[name] = round(statistics.mean(clean), 4) if clean else None  # 有值才算均值，否则 None
        out["per_sample_" + name] = [None if v != v else round(float(v), 4) for v in vals]  # nan 记 None，不伪装成数字
        valid[name] = f"{len(clean)}/{len(vals) or len(rows)}"        # 有效样本数必须上报
        print(f"[ragas] {name} = {out[name]} 有效 {valid[name]} 用时 {round(time.time() - t0)}s")  # 逐指标进度
    out["metric_valid"], out["n_samples"] = valid, len(rows)         # 附加样本口径
    return out                                                       # 返回指标集合
def run(limit=None, mode="hybrid", skip_ragas=False):                # 评估总入口
    rows, tag = collect(limit, mode)                                 # 先作答并落盘
    lat = sorted(r["latency_ms"] for r in rows)                      # 升序便于取分位
    res = {"tag": tag, "mode": mode, "n": len(rows), "errors": sum(1 for r in rows if r["error"]), "p50_ms": lat[len(lat) // 2] if lat else 0, "max_ms": lat[-1] if lat else 0,  # 批次汇总与时延
           "questions": rows, "ragas": {} if skip_ragas else ragas_metrics(rows)}  # 逐题明细 + 指标（可跳过，先看检索效果）
    p = C.RESULTS / f"eval_{tag}.json"                               # 结果文件
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")  # 落盘
    print(f"[eval] 完成 → {p}")                                       # 打印路径
    return res                                                       # 返回
if __name__ == "__main__":                                          # 支持 python -m app.evaluate
    run()                                                           # 执行
