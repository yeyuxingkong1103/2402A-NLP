# 工单编号：人工智能NLP-RAG-Query理解优化
# 文件说明：评估层。用 ragas 计算 faithfulness / answer_relevancy / context_precision / context_recall。
#           踩过的坑（实测，配 DeepSeek 时必看）：
#             1) max_workers 用默认 16 会打满限流，指标被打成 nan —— 故显式降到 EVAL_MAX_WORKERS；
#             2) DeepSeek 不支持 n>1，必须 bypass_n，否则请求直接失败；
#             3) 余额 402 会让整轮静默作废（16 条里 10 条变 nan 都不报错）—— 故结果先落盘再算指标；
#             4) 指标自身有噪声底（约 ±0.05），比较"改前/改后"前必须先量一次噪声底。
import re                                                                       # 剥离引用标记
from typing import Dict, List, Optional                                         # 类型标注

from . import config as C                                                       # 复用集中配置

METRIC_NAMES = ["faithfulness", "answer_relevancy",                              # ragas 四项指标名
                "llm_context_precision_with_reference", "context_recall"]      # 与下方构造顺序一致
# 引用标记形如【招股说明书1 P129】，是工单要求的"回答附带引用"，属于元数据而非内容主张。
_CITATION = re.compile(r"【[^】]*】")                                             # 引用标记


def strip_citations(text: str) -> str:                                          # 去掉引用标记
    """评测前剥离引用标记。

    为什么要剥离（实测）：faithfulness 的判官会把答案拆成独立陈述，逐条核对是否有上下文支撑。
    引用标记【招股说明书1 P129】不是内容主张，但会被拆成一条"无法被上下文支撑"的陈述，
    于是**工单要求的引用功能反而拉低了忠实度指标**。
    实测同一批 10 条答案：带引用标记 faithfulness=0.5237，剥离后=**0.9708**。
    差异几乎全部来自标记本身，所以评测口径统一按"剥离后"计算，并在报告中同时给出两个数字。
    """
    return _CITATION.sub("", text or "").strip()                                # 去掉标记并去首尾空白


def _judge_llm():                                                               # 构造判官大模型
    """用 DeepSeek 作判官：走 OpenAI 兼容接口，并关掉 n>1 的多候选请求。"""
    from openai import OpenAI                                                   # 延迟导入
    from ragas.llms import llm_factory                                          # ragas 的 LLM 工厂
    client = OpenAI(api_key=C.DEEPSEEK_API_KEY, base_url=C.LLM_BASE_URL)        # 复用问答同一把密钥
    kw = {}                                                                     # 需要透传给底层补全调用的参数
    # 坑（实测）：不要把 bypass_n 塞进 llm_factory —— 它会被原样透传给 openai 的
    # Completions.create()，直接报 "unexpected keyword argument 'bypass_n'"，
    # 于是 10 条样本全部失败、四项指标静默变成 null（看起来像"模型不行"，其实是参数用错层）。
    # 本版本 ragas 走 instructor 结构化输出，不再需要 n>1，因此这里不传该参数。
    if C.LLM_REASONING_EFFORT:                                                  # 配了 reasoning_effort 才传
        kw["reasoning_effort"] = C.LLM_REASONING_EFFORT                         # 推理型模型需显式 none，否则思考吃光 max_tokens
    elif "flash" in C.LLM_MODEL.lower() or "reason" in C.LLM_MODEL.lower():     # 模型名像推理型
        kw["reasoning_effort"] = "none"                                         # 关掉思考，避免返回空内容
    return llm_factory(C.LLM_MODEL, provider="openai", client=client, **kw)     # 返回判官


def _embeddings():                                                              # 构造嵌入模型
    """用本地 bge-m3，既省调用费又保证与检索口径一致（都做 L2 归一化）。"""
    from langchain_huggingface import HuggingFaceEmbeddings                     # 延迟导入
    from ragas.embeddings import LangchainEmbeddingsWrapper                     # ragas 适配器
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(                    # 包装成 ragas 可用的嵌入
        model_name=C.EMBED_MODEL_PATH,                                          # 本地路径，不联网
        model_kwargs={"device": C.DEVICE},                                      # 设备
        encode_kwargs={"normalize_embeddings": True}))                          # 与检索口径一致


def evaluate_rows(rows: List[Dict]) -> Dict:                                    # 计算四项指标
    """输入逐题明细（question/answer/ground_truth/contexts），返回四项指标均分。

    单条失败不中断整体（raise_exceptions=False）；最终把 nan 显式转成 None，
    避免把"没算出来"伪装成一个数字。
    """
    if not rows:                                                                # 没有样本
        return {"error": "没有可评估的样本"}                                       # 明确返回
    try:                                                                        # ragas 与依赖可能缺失
        from ragas import EvaluationDataset, SingleTurnSample, evaluate         # ragas 入口
        from ragas.metrics import (AnswerRelevancy, Faithfulness,               # 四项指标
                                   LLMContextPrecisionWithReference,            # 上下文精确率
                                   LLMContextRecall)                            # 上下文召回率
        from ragas.run_config import RunConfig                                  # 并发与超时配置
    except ImportError as exc:                                                  # 缺依赖
        return {"error": f"当前解释器未安装 ragas 依赖：{exc}"}                     # 明确报因
    samples = []                                                                # 样本容器
    for r in rows:                                                              # 逐题构造
        samples.append(SingleTurnSample(user_input=r["question"],               # 用户问题
                                        response=(strip_citations(r.get("answer")) or "（无答案）"),  # 系统回答（剥离引用标记，见 strip_citations）
                                        reference=r.get("ground_truth", ""),    # 标准答案
                                        retrieved_contexts=(r.get("contexts") or ["（无检索结果）"])))  # 检索到的上下文
    judge = _judge_llm()                                                        # 判官
    emb = _embeddings()                                                         # 嵌入
    cfg = RunConfig(max_workers=C.EVAL_MAX_WORKERS,                             # 坑：并发过高会打满限流把指标打成 nan
                    timeout=180, max_retries=3)                                 # 单条超时与重试
    scores = evaluate(EvaluationDataset(samples),                               # 数据集
                      metrics=[Faithfulness(), AnswerRelevancy(strictness=1),   # 忠实度、答案相关性（strictness=1：只生成一个反推问题，省调用也避开 n>1）
                               LLMContextRecall(),                              # 上下文召回率
                               LLMContextPrecisionWithReference()],             # 上下文精确率
                      llm=judge, embeddings=emb,                                # 判官与嵌入
                      run_config=cfg,                                           # 并发受控
                      show_progress=False, raise_exceptions=False)              # 单条失败不中断整体
    df = scores.to_pandas()                                                     # 转 DataFrame
    out: Dict = {}                                                              # 结果
    for name in METRIC_NAMES:                                                   # 逐个指标取均值
        try:                                                                    # 指标列可能缺失
            v = float(df[name].mean())                                          # 求均值
            out[name] = None if v != v else round(v, 4)                         # nan 转 None，不伪装成数字
        except Exception:                                                       # 列不存在
            out[name] = None                                                    # 记为未计算
    out["n_samples"] = len(samples)                                             # 样本数，便于判断结果有效性
    out["valid_samples"] = int(df[METRIC_NAMES[1]].notna().sum()) if METRIC_NAMES[1] in df else 0  # 有效样本数
    return out                                                                  # 返回
