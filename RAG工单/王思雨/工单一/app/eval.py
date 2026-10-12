# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 本模块职责（评估侧）：跑工单 10 问 → RAG 与纯 LLM 双路回答 → RAGAS 四指标 → 出报告。
# 说明：本模块只服务离线评估，不参与在线问答链路。

import json                           # 读写题目集与结果文件
import re                             # 关键事实匹配时的文本归一化
import time                           # 计时
from pathlib import Path              # 路径处理

from . import config as C             # 全局配置
from . import rag                     # 向量化（供 RAGAS 的 embedding 指标使用）
from . import llm as L                # 流式生成（与线上同路径）
from . import retrieval as R          # 在线问答链路


def _norm(text: str) -> str:                                          # 关键事实匹配前的归一化
    """去掉空格与千分位逗号、统一百分号，避免"6,464.51"与"6464.51"被误判为不匹配。"""
    return re.sub(r"[\s,，]", "", text or "").replace("％", "%")        # 归一化


def _hits(answer: str, expect: list[str]) -> list[str]:               # 关键事实命中检测
    """返回答案中命中的关键事实列表；工单 10 问均可用少量关键数字/实体判定对错。"""
    a = _norm(answer)                                                 # 归一化答案
    return [e for e in expect if _norm(e) in a]                       # 逐个比对，返回命中项


def load_questions() -> list[dict]:                                   # 读取工单题目集
    """从 eval/questions.json 读取 10 个问题及其原文标准答案。"""
    p = C.EVAL_DIR / "questions.json"                                 # 题目文件路径
    return json.loads(p.read_text(encoding="utf-8"))["questions"]     # 返回 questions 数组


class _BgeEmbeddings:                                                 # 给 RAGAS 用的向量接口适配器
    """把本地 bge-m3 包装成 RAGAS/LangChain 期望的 embeddings 接口（answer_relevancy 需要）。"""

    def embed_documents(self, texts):                                 # 批量编码文档
        return rag.embed_texts(list(texts)).tolist()                  # 复用离线侧同一模型，保证一致

    def embed_query(self, text):                                      # 单条查询编码
        return rag.embed_texts([text])[0].tolist()                    # 返回列表形式


def _judge():                                                         # 构造 RAGAS 评判器
    """用 DeepSeek 作评判 LLM、本地 bge-m3 作评判嵌入，避免依赖 OpenAI 官方接口。

    实测两种配置（各跑 2 个样本）：
      开推理 + max_tokens=8192：85.8s，其中 1 个样本仍报 IncompleteOutputException → 分数为 NaN；
      关推理 + max_tokens=2048：26.3s，两个样本都正常出分。
    评判任务本质是"按规则打标签"，不需要长推理；开推理反而因 reasoning_tokens 吃光额度而失败。
    故判官统一关闭推理。
    """
    import openai                                                     # 延迟导入
    from ragas.llms import llm_factory                                # RAGAS 0.4 的 LLM 工厂
    from ragas.embeddings import LangchainEmbeddingsWrapper           # 包装器
    client = openai.OpenAI(base_url=C.LLM_BASE_URL, api_key=C.LLM_API_KEY,
                           timeout=C.LLM_TIMEOUT)                     # 指向 DeepSeek，带超时
    judge_llm = llm_factory(C.LLM_MODEL, client=client,               # 构建评判模型
                            max_tokens=C.EVAL_JUDGE_MAX_TOKENS,       # 输出额度
                            reasoning_effort="none")                  # 关闭长推理，避免空输出
    judge_emb = LangchainEmbeddingsWrapper(_BgeEmbeddings())          # 构建评判嵌入（本地 bge-m3）
    return judge_llm, judge_emb                                       # 返回两者


def _rag_stream(q: str, use_rerank: bool = True) -> tuple[str, list[str], dict]:   # 走流式链路并测首字延迟
    """调用与线上一致的流式问答，返回 (完整答案, 引用片段文本, 耗时明细)。

    之所以用流式而不是非流式：线上前端走的就是 SSE，首字延迟才是用户感知的响应时间，
    测试必须与线上路径一致，避免"测的是一套、跑的是另一套"。
    """
    t0 = time.time()                                                  # 计时起点
    ret = R.retrieve(q, use_rewrite="auto", use_rerank=use_rerank)    # 检索（含 Query 理解）
    t_ret = time.time() - t0                                          # 检索耗时
    chunks_out, first = [], None                                      # 答案片段与首字时刻
    msgs = R.build_prompt(q, ret["contexts"])                         # 组装提示词
    try:                                                              # 生成阶段容错
        for piece in L.llm_chat(msgs, max_tokens=450, temperature=0.0,
                                stream=True):   # 流式生成（与线上一致，0 温度）
            if first is None:                                         # 第一个增量
                first = time.time()                                   # 记录首字时刻
            chunks_out.append(piece)                                  # 累积
    except Exception as exc:                                          # LLM 不可用
        chunks_out.append(R._fallback_answer(q, ret["contexts"], exc))  # 兜底摘录
    total = time.time() - t0                                          # 端到端
    lat = {**ret["timing"], "first_token": round((first - t0) if first else total, 3),   # 首字延迟
           "generate": round(total - t_ret, 3), "total": round(total, 3)}                # 生成与总计
    # 关键：回传的是**带页码标注**的片段文本（与模型实际看到的完全一致）。
    # 若传裸文本，模型答案里的"（依据：第N页）"会被 RAGAS 判为无依据，faithfulness 被系统性低估。
    return "".join(chunks_out), R.format_contexts(ret["contexts"]), lat   # 返回三项


def _run_questions(use_rerank: bool = True) -> list[dict]:            # 跑完 10 问并采集双路回答
    """逐题执行 RAG 问答与纯 LLM 问答，记录答案、引用片段、首字与总耗时。"""
    rows = []                                                         # 结果行
    for item in load_questions():                                     # 遍历工单 10 问
        q = item["question"]                                          # 问题文本
        print(f"[评估] 正在处理 id={item['id']}: {q[:30]}…")           # 进度提示
        try:                                                          # RAG 路径容错
            rag_ans, ctxs, lat = _rag_stream(q, use_rerank)           # 走线上同款流式链路
        except Exception as exc:                                      # 问答失败
            rag_ans, ctxs, lat = f"（RAG 失败：{type(exc).__name__}）", [], {"total": 0.0}   # 记录失败
        try:                                                          # 纯 LLM 路径容错
            llm_ans = R.answer_llm_only(q)["answer"]                  # 不传上下文
        except Exception as exc:                                      # 调用失败
            llm_ans = f"（纯 LLM 失败：{type(exc).__name__}）"          # 记录失败
        exp = item.get("expect", [])                              # 关键事实校验项
        rows.append({"id": item["id"], "question": q,                 # 组装结果行
                     "expect": exp,                                   # 校验项
                     "rag_hits": _hits(rag_ans, exp),                 # RAG 命中
                     "llm_hits": _hits(llm_ans, exp),                 # 纯 LLM 命中
                     "ground_truth": item["ground_truth"],            # 原文标准答案
                     "source_pages": item.get("source_pages", []),    # 答案所在页
                     "rag_answer": rag_ans,                           # RAG 回答
                     "llm_only_answer": llm_ans,                      # 纯 LLM 回答
                     "retrieved_contexts": ctxs,                      # 检索到的片段（送 RAGAS）
                     "n_contexts": len(ctxs),                         # 片段数
                     "latency": lat})                                 # 耗时明细
    return rows                                                       # 返回全部结果行


def _ragas_scores(rows: list[dict]) -> dict:                          # 计算 RAGAS 四指标
    """用 RAGAS 计算 faithfulness / answer_relevancy / context_precision / context_recall。"""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate   # RAGAS 0.4 数据集接口
    from ragas.metrics import (faithfulness, answer_relevancy,        # 四指标实例
                               context_precision, context_recall)
    samples = [SingleTurnSample(user_input=r["question"], response=r["rag_answer"],
                                retrieved_contexts=r["retrieved_contexts"] or ["（无检索结果）"],
                                reference=r["ground_truth"]) for r in rows]   # 构造样本（空上下文需占位）
    judge_llm, judge_emb = _judge()                                   # 构造评判器
    result = evaluate(dataset=EvaluationDataset(samples=samples),     # 执行评估
                      metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                      llm=judge_llm, embeddings=judge_emb,             # 指定评判模型
                      raise_exceptions=False, show_progress=True)      # 单指标失败不中断整体
    try:                                                              # 优先取结构化分数
        df = result.to_pandas()                                       # 转 DataFrame
        return {"overall": {m: round(float(df[m].mean()), 4) for m in
                            ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
                            if m in df.columns},                       # 四个指标均值
                "per_question": json.loads(df.to_json(orient="records", force_ascii=False))}   # 逐题分数
    except Exception:                                                 # to_pandas 不可用
        return {"overall": dict(result), "per_question": []}           # 退回原始字典


def _save(rows: list[dict], scores: dict, seconds: float, use_rerank: bool) -> Path:   # 保存评估结果
    """把对比结果与 RAGAS 分数写入 eval/results/latest.json 与带时间戳的历史文件。"""
    lat = [r["latency"].get("total", 0) for r in rows]                # 各题端到端耗时
    ttft = [r["latency"].get("first_token", 0) for r in rows]         # 各题首字延迟
    lat_sorted, ttft_sorted = sorted(lat), sorted(ttft)               # 升序，便于取分位
    n_expect = sum(len(r.get("expect", [])) for r in rows) or 1       # 关键事实总数
    rag_hit = sum(len(r.get("rag_hits", [])) for r in rows)           # RAG 命中数
    llm_hit = sum(len(r.get("llm_hits", [])) for r in rows)           # 纯 LLM 命中数
    summary = {                                                       # 汇总信息
        "n": len(rows),                                               # 题目数
        "use_rerank": use_rerank,                                     # 是否启用重排
        "avg_latency": round(sum(lat) / len(lat), 3) if lat else 0.0,  # 平均总耗时
        "p50_latency": round(lat_sorted[len(lat) // 2], 3) if lat else 0.0,        # 总耗时 P50
        "max_latency": round(max(lat), 3) if lat else 0.0,             # 最大耗时
        "avg_first_token": round(sum(ttft) / len(ttft), 3) if ttft else 0.0,       # 平均首字延迟
        "p50_first_token": round(ttft_sorted[len(ttft) // 2], 3) if ttft else 0.0, # 首字 P50
        "under_3s_rate": round(sum(1 for x in lat if x <= 3.0) / len(lat), 4) if lat else 0.0,  # 总耗时达标率
        "ttft_under_3s_rate": round(sum(1 for x in ttft if x <= 3.0) / len(ttft), 4) if ttft else 0.0,  # 首字达标率
        "key_fact_total": n_expect,                                   # 关键事实总数
        "rag_key_fact_rate": round(rag_hit / n_expect, 4),            # RAG 关键事实命中率
        "llm_only_key_fact_rate": round(llm_hit / n_expect, 4),       # 纯 LLM 关键事实命中率
        "eval_seconds": round(seconds, 1)}                             # 评估总耗时
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),     # 生成时间
               "summary": summary, "ragas": scores, "items": rows}     # 完整结果
    out_dir = C.EVAL_DIR / "results"                                  # 结果目录
    out_dir.mkdir(parents=True, exist_ok=True)                        # 确保存在
    text = json.dumps(payload, ensure_ascii=False, indent=2)          # 序列化
    (out_dir / "latest.json").write_text(text, encoding="utf-8")      # 固定名，供接口读取
    (out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(text, encoding="utf-8")  # 历史留档
    return out_dir / "latest.json"                                    # 返回路径


def run_eval(task_id: str, tasks: dict, use_rerank: bool = True) -> None:   # 评估入口（后台任务）
    """供 /api/eval/run 调用的后台函数：跑题 → 算 RAGAS → 落盘，并持续更新任务状态。"""
    t0 = time.time()                                                  # 计时
    tasks[task_id].update(status="running", message="正在执行 10 题问答…")   # 更新状态
    try:                                                              # 整体容错
        rows = _run_questions()                                       # 跑题
        tasks[task_id].update(message="问答完成，正在计算 RAGAS 指标…")    # 更新状态
        try:                                                          # RAGAS 单独容错
            scores = _ragas_scores(rows)                              # 计算四指标
        except Exception as exc:                                      # 指标计算失败
            scores = {"error": f"{type(exc).__name__}: {exc}"}        # 记录错误但保留对比结果
        path = _save(rows, scores, time.time() - t0, use_rerank)      # 落盘
        tasks[task_id].update(status="done", message=f"评估完成，结果已写入 {path.name}",
                              seconds=round(time.time() - t0, 1))     # 完成
    except Exception as exc:                                          # 未预期异常
        tasks[task_id].update(status="failed", message=f"{type(exc).__name__}: {exc}")   # 标记失败
