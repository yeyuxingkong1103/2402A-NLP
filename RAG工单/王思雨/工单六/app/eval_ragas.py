# 工单编号：人工智能NLP-RAG-混合检索任务
# 文件说明：RAGAS 评估。计算 faithfulness / answer_relevancy / context_precision / context_recall，
#           并用关键词判分给出验收口径的准确率，全部逐题明细先落盘再算指标（防限流/402 丢数据）。
import json, re, time                                                              # 标准库
from . import config as C                                                           # 全局配置
from . import rag_core as R                                                         # 复用问答链路

_CTX = re.compile(r"【[^】]*】")                                                      # 引用标记（评测前必须剥离，否则拉低 faithfulness）
_NEUTRAL = re.compile(r"[,\s，]")                                                    # 千分位逗号与空白
METRICS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]  # 四个指标列名

def _flat(s: str) -> str:                                                           # 判分归一化
    """去掉千分位逗号与空白，使 6,464.51 与 6464.51、5,000 与 5000 视为同一结果。"""
    return _NEUTRAL.sub("", s or "")                                                # 返回归一化串

def _judge():                                                                       # RAGAS 判官
    """DeepSeek 充当判官；不要传 bypass_n，ragas 走 instructor 结构化输出，多传会报 unexpected keyword。"""
    from openai import OpenAI                                                       # 延迟导入
    from ragas.llms import llm_factory                                              # ragas 0.4 的 LLM 工厂
    cli = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL)               # 复用同一套密钥与基址
    kw = {"reasoning_effort": C.LLM_REASONING_EFFORT} if C.LLM_REASONING_EFFORT else {}  # 显式配置才传
    if not kw and ("flash" in C.LLM_MODEL.lower() or "reason" in C.LLM_MODEL.lower()):  # 推理型模型
        kw["reasoning_effort"] = "none"                                             # 不关思考会返回空内容
    return llm_factory(C.LLM_MODEL, provider="openai", client=cli, **kw)            # 返回判官

def _embeddings():                                                                  # RAGAS 嵌入
    """与检索同口径的本地 bge-m3（L2 归一化），避免判官嵌入与检索嵌入不一致导致指标虚高或虚低。"""
    from langchain_huggingface import HuggingFaceEmbeddings                          # 延迟导入
    from ragas.embeddings import LangchainEmbeddingsWrapper                          # ragas 包装器
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(                        # 包装本地模型
        model_name=C.EMBED_MODEL_PATH, model_kwargs={"device": C.DEVICE},           # 本地路径与设备
        encode_kwargs={"normalize_embeddings": True}))                              # 与检索一致的归一化

def _correct(ans: str, q: dict) -> bool:                                            # 关键词判分
    """expect 全含、expect_any 每组至少一项出现、expect_not 负样本一个不出现，三者齐备才算对。

    expect_any 用于同一事实的多种正确表述：如"深圳证券交易所"与"创业板"指的是同一个板块，
    没有它会把"表述正确但用词不同"的答案误判成错——本套题集实测被误判过一次（q17）。
    expect_not 专抓"多答无关内容"与"答成另一家公司"。
    """
    a = _flat(ans)                                                                  # 归一化答案
    ok_must = all(_flat(e) in a for e in q.get("expect", []))                       # 必需项全含
    ok_any = all(any(_flat(x) in a for x in grp) for grp in q.get("expect_any", []))  # 每组同义表述至少中一个
    ok_not = not any(_flat(n) in a for n in q.get("expect_not", []))                # 负样本一个都不出现
    return ok_must and ok_any and ok_not                                            # 三者齐备

def _load(limit=None) -> list:                                                      # 读题集
    """题集支持 list 或 {questions: [...]} 两种结构，便于外部直接替换。"""
    data = json.loads(C.EVAL_FILE.read_text(encoding="utf-8"))                      # 读文件
    rows = data.get("questions", data) if isinstance(data, dict) else data          # 兼容两种结构
    return rows[:limit] if limit else rows                                          # 按需截断

def collect(limit=None) -> list:                                                    # 逐题作答
    """跑完整问答链路收集答案与检索上下文；同一会话的题按顺序提问以验证多轮指代消解。"""
    rows, seen = _load(limit), set()                                                # 题集与已初始化会话
    out = []                                                                        # 结果
    t0 = time.time()                                                                # 预热计时
    R.retrieve("预热查询", "hybrid", 3, rerankers=C.DEFAULT_RERANKERS)                 # 预热检索侧：编码/Milvus/BM25/融合/重排打分
    try:                                                                            # 预热大模型连接
        R.llm_chat([{"role": "user", "content": "hi"}], max_tokens=1, timeout=30)     # 付掉 DNS+TLS 握手
    except Exception:                                                               # 预热失败不影响评估
        pass                                                                        # 忽略
    print(f"[评估] 预热耗时 {time.time() - t0:.1f}s（不计入逐题耗时；线上服务在启动时就完成了这一步）", flush=True)
    for i, q in enumerate(rows, 1):                                                 # 逐题
        sid = q.get("session") or f"eval-{q['id']}"                                 # 会话号：显式指定则共用
        if sid not in seen:                                                         # 该会话首次出现
            try:                                                                    # 清掉上一轮的残留历史
                R.redis_client().delete(f"gd6:hist:{sid}")                          # 避免多次评估互相污染
            except Exception:                                                       # Redis 不可用
                pass                                                                # 无视，退化为单轮
            seen.add(sid)                                                           # 标记已初始化
        r = R.answer(q["question"], session_id=sid, mode=q.get("mode", "hybrid"),    # 走与线上一致的链路
                     fusion=q.get("fusion", "weighted"),                            # 融合算法
                     rerankers=q.get("rerankers", C.DEFAULT_RERANKERS),                # 级联重排
                     use_cache=False)                                               # 评估必须绕开缓存，否则测不出真实耗时
        ans = _CTX.sub("", r["answer"])                                             # 剥离引用标记再判分
        ok = _correct(ans, q)                                                       # 关键词判分
        out.append({**q, "answer": ans, "contexts": r["contexts"],                  # 保留题目与答案
                    "latency_ms": r["latency_ms"], "degraded": r["degraded"],       # 耗时与降级
                    "retrieval_query": r["retrieval_query"], "correct": ok})        # 消解后问句与判分
        print(f"  [{i}/{len(rows)}] {'✓' if ok else '✗'} {r['latency_ms']:>5}ms "   # 实时进度
              f"{q['id']} {q['question'][:32]}", flush=True)                        # 题号与题干
    return out                                                                      # 返回明细

def _ragas(rows: list) -> dict:                                                     # RAGAS 指标
    """单条失败不中断整体；nan 显式转 None，绝不把无效值伪装成数字。"""
    try:                                                                            # 依赖可能缺失
        from ragas import EvaluationDataset, SingleTurnSample, evaluate              # ragas 0.4 API
        from ragas.metrics import (AnswerRelevancy, Faithfulness,                    # 指标类
                                   LLMContextPrecisionWithReference, LLMContextRecall)  # 上下文类
        from ragas.run_config import RunConfig                                       # 并发控制
    except Exception as exc:                                                         # 缺依赖
        return {"error": f"当前解释器未安装 ragas 依赖：{type(exc).__name__}: {exc}"}  # 明确报因
    samples = [SingleTurnSample(user_input=r["question"], response=r["answer"] or "（无答案）",  # 逐条建样本
                                reference=r.get("ground_truth", ""),                # 标准答案
                                retrieved_contexts=r["contexts"] or ["（无检索结果）"])  # 检索上下文
               for r in rows]                                                       # 覆盖全部题
    if not samples:                                                                 # 无样本
        return {"error": "没有可评估的样本"}                                           # 明确报因
    # 逐指标串行评估：一次只测一个指标。四个指标一起跑会让并发请求数直接翻四倍，
    # 更容易触发限流并把一部分样本静默打成 nan；分开跑还能分别上报各指标的有效样本数。
    plan = [("faithfulness", Faithfulness()),                                       # 忠实度
            ("answer_relevancy", AnswerRelevancy(strictness=1)),                    # strictness=1 只生成一个反推问题
            ("context_recall", LLMContextRecall()),                                 # 召回率口径
            ("llm_context_precision_with_reference", LLMContextPrecisionWithReference())]  # 精确率
    ds, llm, emb = EvaluationDataset(samples), _judge(), _embeddings()               # 数据集 / 判官 / 嵌入（只建一次）
    out, cols = {}, {}                                                              # 指标结果与逐题列
    for name, metric in plan:                                                       # 逐个指标
        try:                                                                        # 该指标可能整体失败
            res = evaluate(ds, metrics=[metric], llm=llm, embeddings=emb,            # 单指标评估
                           run_config=RunConfig(max_workers=C.EVAL_MAX_WORKERS,      # 默认 16 并发会打满限流
                                                timeout=180, max_retries=3),         # 超时与重试
                           show_progress=False, raise_exceptions=False)              # 单条失败不中断整体
            series = res.to_pandas()[name]                                           # 该指标的逐题得分
            v = float(series.mean())                                                 # 均值
            out[name] = None if v != v else round(v, 4)                              # nan 显式转 None，不伪装成数字
            out.setdefault("metric_valid", {})[name] = int(series.notna().sum())     # 该指标各自的有效样本数
            cols[name] = series                                                      # 供后续拼逐题明细
            print(f"  [RAGAS] {name}: 有效 {out['metric_valid'][name]}/{len(series)}，均值 {out[name]}", flush=True)
        except Exception as exc:                                                    # 该指标失败
            print(f"  [RAGAS] {name} 执行失败：{type(exc).__name__}: {exc}", flush=True)  # 明确报因
            out[name] = None                                                        # 记为未计算
            out.setdefault("metric_valid", {})[name] = 0                             # 有效率记 0
    out["n_samples"] = len(samples)                                                 # 样本数
    out["valid_samples"] = out.get("metric_valid", {}).get("answer_relevancy", 0)    # 有效样本数，远小于样本数即被限流
    if cols:                                                                        # 有逐题列
        import pandas as pd                                                         # 延迟导入
        out["per_sample"] = json.loads(pd.DataFrame(cols).to_json(orient="records", force_ascii=False))  # 逐题得分，便于核查哪几条失败
    return out                                                                      # 返回

def run(limit=None) -> dict:                                                        # 评估主流程
    """先逐题作答并落盘，再补算 RAGAS 指标并覆盖保存；返回准确率/召回率/响应时间与指标。"""
    t0 = time.time()                                                                # 总计时
    rows = collect(limit)                                                           # 收集
    n = len(rows)                                                                   # 题数
    acc = round(sum(1 for r in rows if r["correct"]) / n, 4) if n else None          # 关键词判分准确率
    lat = sorted(r["latency_ms"] for r in rows)                                     # 耗时排序
    base = {"ts": time.strftime("%Y%m%d_%H%M%S"), "n": n,                           # 时间戳与题数
            "accuracy": acc,                                                        # 验收口径准确率
            "latency": {"p50": lat[len(lat) // 2] if lat else None,                  # 中位响应时间
                        "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,  # 95 分位
                        "max": lat[-1] if lat else None,                            # 最大
                        "over_3s": sum(1 for x in lat if x > 3000),                 # 超 3 秒的题数，如实统计不隐去
                        "target_ms": int(C.TARGET_LATENCY * 1000)},                 # 验收目标
            "rows": rows}                                                           # 逐题明细
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)                                 # 确保目录
    path = C.RESULT_DIR / f"eval_{base['ts']}.json"                                 # 结果文件
    path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")  # 先落盘，防后续失败丢数据
    base["ragas"] = _ragas(rows)                                                    # 再算指标
    rec = (base["ragas"] or {}).get("context_recall")                               # 用 context_recall 作为召回率口径
    base["recall"] = rec                                                            # 召回率
    base["mapped"] = {"accuracy": acc, "recall": rec,                               # 验收口径映射
                      "note": "准确率=关键词判分(expect 全含且 expect_not 不出现)；召回率=ragas context_recall"}  # 口径说明
    base["targets"] = {"accuracy": C.TARGET_ACCURACY, "recall": C.TARGET_RECALL,     # 目标值
                       "met_accuracy": (acc or 0) >= C.TARGET_ACCURACY,             # 准确率是否达标
                       "met_recall": (rec or 0) >= C.TARGET_RECALL}                 # 召回率是否达标
    base["elapsed_s"] = round(time.time() - t0, 1)                                  # 总耗时
    path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")  # 覆盖保存
    base["result_file"] = str(path)                                                 # 回带文件路径
    return base                                                                     # 返回

if __name__ == "__main__":                                                          # 允许命令行直接跑
    import sys                                                                      # 取参数
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None                           # 可选限制题数
    r = run(limit=lim)                                                              # 执行
    print(json.dumps({k: v for k, v in r.items() if k != "rows"},                   # 打印汇总
                     ensure_ascii=False, indent=2))                                 # 不打印逐题明细
