# -*- coding: utf-8 -*-
"""RAGAS 评测模块：跑 15 条评测集，用四个指标打分，结果落盘 data/ragas_result.json。

依赖 ragas 0.4.3。该版本里有两套指标 API，必须区分清楚：
  - ragas.metrics.collections.*  是新式指标，但它**不是** Metric 的子类，
    传给 evaluate() 会被拒收（报 "All metrics must be initialised metric objects"）；
  - ragas.metrics.*（如 Faithfulness）是旧式指标，**是** Metric 的子类，
    evaluate() 只认这一套，且它的 llm 参数要的是 BaseRagasLLM。
因此下面用「旧式指标 + LangchainLLMWrapper / LangchainEmbeddingsWrapper」的组合。
"""

import json                                    # 读写评测集与结果文件
import math                                    # 判断 NaN，指标算不出来时不污染汇总
import time                                    # 生成结果文件的时间戳
from pathlib import Path                       # 拼接数据文件路径
from typing import List                         # 类型注解：列表

import config                                  # 读模型名、API Key 与地址
import db                                      # 取 Redis 客户端，用于清评测缓存
import db_user                                 # 建评测专用会话
import memory                                  # 清评测用户的短期与长期记忆
import rag                                     # 跑完整问答链路拿 answer
import retrieval                               # 单独检索拿 retrieved_contexts
import vector_store                            # BGE-m3 向量化
from logger import get_logger                   # 日志工具

logger = get_logger("eval_ragas")               # 创建本模块的 logger 实例

BASE_DIR = Path(__file__).resolve().parent              # 项目根目录
EVAL_SET = BASE_DIR / "data" / "eval_dataset.json"      # 评测集文件
EVAL_OUT = BASE_DIR / "data" / "ragas_result.json"      # 评测结果文件
EVAL_USER_ID = 9901                             # 评测专用用户，避免污染真实用户数据
EVAL_ROLE_ID = 1                                # 固定使用电力维修专家角色
TOP_K = 5                                       # 每条问题取前 5 个上下文
# 四个指标名，同时用作出参顺序与汇总键名
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load_eval_set(path=None) -> List[dict]:
    """读评测集文件，逐条校验 question 与 ground_truth 都不能为空。"""
    target = Path(path) if path else EVAL_SET            # 允许调用方指定路径
    if not target.exists():                              # 文件不存在
        raise FileNotFoundError(f"评测集不存在：{target}")   # 立即报错，不静默跳过
    rows = json.loads(target.read_text(encoding="utf-8"))       # 读文件并解析
    if not isinstance(rows, list) or not rows:            # 必须是非空列表
        raise ValueError("评测集必须是非空列表")            # 报错
    for i, row in enumerate(rows, 1):                     # 逐条校验
        if not str(row.get("question", "")).strip():      # 问题不能为空
            raise ValueError(f"第 {i} 条缺少 question")     # 报错并指出第几条
        if not str(row.get("ground_truth", "")).strip():  # 参考答案不能为空
            raise ValueError(f"第 {i} 条缺少 ground_truth")   # 报错并指出第几条
    logger.info("评测集加载完成：%d 条", len(rows))         # 记录条数
    return rows                                           # 返回评测集


def build_llm():
    """构建 RAGAS 的评估 LLM：用 DeepSeek 的 OpenAI 兼容接口包成 RAGAS 可用的 LLM。"""
    from langchain_openai import ChatOpenAI            # 延迟导入，缺依赖时不影响模块加载
    from ragas.llms import LangchainLLMWrapper         # 包装成旧式 BaseRagasLLM
    chat = ChatOpenAI(model=config.LLM_MODEL,          # 模型名，来自 .env
                      base_url=config.LLM_BASE_URL,    # DeepSeek 的 OpenAI 兼容地址
                      api_key=config.LLM_API_KEY,      # API Key，只从 .env 读，不写进代码
                      temperature=0,                   # 评测要可复现，温度固定为 0
                      timeout=120,                     # 单次请求 120 秒超时，长跑时避免被掐断
                      max_retries=3)                   # 网络抖动自动重试 3 次
    return LangchainLLMWrapper(chat)                   # 返回 RAGAS 可用的封装对象


class BgeM3Embeddings:                                 # 把本项目的 BGE-m3 包装给 RAGAS 用
    """按 langchain_core.embeddings.Embeddings 的接口，内部调 vector_store.embed_texts。

    只实现接口需要的两个方法，不真的继承 LangChain 的基类，
    这样没装 langchain 时本模块依然能 import（真正用到时才延迟导入）。
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化：调用项目自己的向量化接口，只取稠密向量。"""
        dense, _ = vector_store.embed_texts(list(texts))   # 返回 (稠密, 稀疏) 两个列表
        return dense                                       # RAGAS 只需要稠密部分

    def embed_query(self, text: str) -> List[float]:
        """单条向量化：复用批量接口后取第一条。"""
        dense, _ = vector_store.embed_texts([text])        # 单条也走批量接口，逻辑统一
        return dense[0]                                    # 返回该条的稠密向量


def build_embeddings():
    """构建 RAGAS 的评估 Embedding：包装本地 BGE-m3，不联网。"""
    from ragas.embeddings import LangchainEmbeddingsWrapper   # 包装成 RAGAS 的 Embedding
    return LangchainEmbeddingsWrapper(BgeM3Embeddings())      # 传入自定义嵌入实现


def clear_eval_cache() -> int:
    """清掉评测用户的答案缓存，保证每次评测都真跑检索与生成，而不是复用上次答案。"""
    try:                                                   # 清缓存失败不影响评测继续
        client = db.get_redis_client()                     # 取 Redis 客户端
        pattern = f"cache:rag:answer:{EVAL_USER_ID}:*"     # 只扫评测用户的答案缓存键
        keys = list(client.scan_iter(match=pattern, count=200))   # 用 SCAN 分批扫，不用 KEYS
        if keys:                                           # 有缓存才删
            client.delete(*keys)                           # 批量删除
        logger.info("评测缓存已清理：%d 个键", len(keys))   # 记录清理条数
        return len(keys)                                   # 返回删除条数
    except Exception as exc:                               # Redis 不可用
        logger.warning("清理评测缓存失败（不影响评测继续）：%s", exc)   # 记录告警
        return 0                                           # 返回 0


def reset_eval_memory() -> None:
    """清空评测用户的短期与长期记忆，保证每条问题独立、不被前一条的问答带偏。

    为什么必须清：ask() 的 history 与长期记忆都是按 user_id 读的，不是按会话读的。
    如果不清，第 2 条问题就会带着第 1 条的问答去做查询改写，
    实测第 2 条的改写结果里混进了第 1 条的内容，faithfulness 直接被拉到 0。
    单轮评测要求每条问题独立作答，所以每条之前都清一次。
    """
    try:                                                   # 短期记忆在 Redis
        memory.clear_short_memory(EVAL_USER_ID)            # 清空该用户的短期记忆
    except Exception as exc:                               # 清不掉不影响评测继续
        logger.warning("清短期记忆失败（不影响评测）：%s", exc)   # 记录告警
    try:                                                   # 长期记忆在 Milvus
        client = memory.get_memory_client()                # 取长期记忆客户端
        name = config.MEMORY_COLLECTION                    # 集合名，来自配置
        client.delete(collection_name=name, filter=f"user_id == {EVAL_USER_ID}")   # 按用户删
        client.refresh_load(name)                          # 刷新已加载快照，否则删完仍能搜到
    except Exception as exc:                               # 清不掉不影响评测继续
        logger.warning("清长期记忆失败（不影响评测）：%s", exc)   # 记录告警


def collect_samples(eval_set: List[dict]) -> List[dict]:
    """逐条跑一遍完整 RAG，收集系统回答与检索到的上下文原文。"""
    samples = []                                           # 收集好的样本
    for i, row in enumerate(eval_set, 1):                  # 逐条评测
        question = row["question"]                         # 当前问题
        logger.info("评测 %d/%d：%s", i, len(eval_set), question[:30])   # 打印进度
        reset_eval_memory()                                # 清记忆，保证本条独立作答
        contexts = retrieval.retrieve(question, top_k=TOP_K)             # 单独检索拿上下文
        conv_id = db_user.create_conversation(EVAL_USER_ID, EVAL_ROLE_ID,   # 每条新建会话
                                              f"RAGAS评测 {i}")              # 会话标题带序号
        result = rag.ask(question, EVAL_USER_ID, EVAL_ROLE_ID, conv_id,   # 跑完整问答链路
                         stream=False)                                    # 非流式拿完整答案
        samples.append({                                   # 组装一条样本
            "question": question,                          # 问题
            "answer": result["answer"],                    # 系统回答
            "ground_truth": row["ground_truth"],           # 参考答案（人工摘录）
            "contexts": [c["text"] for c in contexts],     # 检索到的上下文原文列表
        })                                                 # 样本组装结束
    logger.info("样本收集完成：%d 条", len(samples))        # 记录条数
    return samples                                         # 返回样本列表


def _run_ragas(samples: List[dict]) -> List[dict]:
    """调 RAGAS 跑四个指标，返回与样本一一对应的分数列表。"""
    from ragas import evaluate, EvaluationDataset          # RAGAS 入口与数据集容器
    from ragas.metrics import (Faithfulness, AnswerRelevancy,   # 忠实度、答案相关度
                               ContextPrecision, ContextRecall)  # 上下文精确率、上下文召回率
    llm = build_llm()                                      # 评估 LLM
    embeddings = build_embeddings()                        # 评估 Embedding
    rows = [dict(user_input=s["question"], response=s["answer"],   # 问题与系统回答
                 retrieved_contexts=s["contexts"],                 # 检索上下文
                 reference=s["ground_truth"])                      # 参考答案
            for s in samples]                              # 逐条转成 RAGAS 要的字段名
    dataset = EvaluationDataset.from_list(rows)            # 0.4.3 的 from_list 收 dict 列表
    metrics = [Faithfulness(llm=llm),                      # 忠实度：回答是否忠于上下文
               AnswerRelevancy(llm=llm, embeddings=embeddings,   # 答案相关度：需要向量
                               strictness=1),                  # 必须设 1，DeepSeek 只支持 n=1
               ContextPrecision(llm=llm),                  # 上下文精确率：检索是否精准
               ContextRecall(llm=llm)]                     # 上下文召回率：是否漏掉依据
    outcome = evaluate(dataset=dataset, metrics=metrics, llm=llm,   # 执行评测
                       embeddings=embeddings, show_progress=False)  # 关进度条，日志自己打
    scores = outcome.scores                                # 每条样本一行分数
    if len(scores) != len(samples):                        # 条数必须对齐才能配对
        raise RuntimeError(f"RAGAS 返回 {len(scores)} 行，与样本 {len(samples)} 条不一致")
    return scores                                          # 返回分数列表


def _metric_value(value):
    """把 RAGAS 的原始分数规整成 JSON 安全的数；算不出来（空/NaN/非数）一律返回 None。"""
    try:                                                   # 类型转换可能抛异常
        num = float(value)                                 # 先转成浮点
    except (TypeError, ValueError):                        # 不是数字
        return None                                        # 视为"这一项没算出来"
    return None if math.isnan(num) else round(num, 4)      # NaN 同样视为没算出来


def _metric_mean(values: list):
    """对某个指标求均分，跳过算不出来的样本；一条可用值都没有时返回 None。"""
    usable = [v for v in values if v is not None]          # 只保留算出来的值
    if not usable:                                         # 全都没算出来
        return None                                        # 返回 None，不编一个 0 冒充分数
    return round(sum(usable) / len(usable), 4)             # 对可用样本求均分


def run_evaluation() -> dict:
    """跑完整评测：加载评测集 → 逐条问答 → RAGAS 打分 → 汇总均分与明细。"""
    eval_set = load_eval_set()                             # 读评测集
    clear_eval_cache()                                     # 清掉上次的答案缓存
    samples = collect_samples(eval_set)                    # 逐条跑 RAG，收集样本
    scores = _run_ragas(samples)                           # 跑四个指标
    details = []                                           # 逐条明细
    for sample, score in zip(samples, scores):             # 样本与分数按顺序配对
        details.append({                                   # 组装一条明细
            "question": sample["question"],                # 问题
            "answer": sample["answer"],                    # 系统回答
            "ground_truth": sample["ground_truth"],        # 参考答案
            # 某项算不出来就写 null，绝不用 0 冒充"得了 0 分"——两者含义完全不同
            **{name: _metric_value(score.get(name)) for name in METRIC_NAMES},
        })                                                 # 明细组装结束
    summary = {name: _metric_mean([d[name] for d in details])   # 每个指标对可用样本求均分
               for name in METRIC_NAMES}                   # 逐指标计算
    for name in METRIC_NAMES:                              # 逐指标打印
        skipped = sum(1 for d in details if d[name] is None)   # 该项有多少条没算出来
        logger.info("指标均分 %-18s = %-8s（跳过 %d 条算不出的）",
                    name, summary[name], skipped)          # 打印均分与跳过条数
        if skipped:                                        # 有跳过的必须显式提醒
            logger.warning("指标 %s 有 %d 条没算出来，均分只覆盖其余样本；"
                           "常见原因是评估模型额度不足或网络超时，请看上面的异常日志",
                           name, skipped)                  # 提醒均分的覆盖范围不完整
    return {"summary": summary, "details": details}        # 返回汇总与明细


def save_result(result: dict, out_path=None) -> Path:
    """把评测结果写成 JSON，含时间戳、模型名、指标汇总与逐条明细。"""
    target = Path(out_path) if out_path else EVAL_OUT      # 允许调用方指定路径
    target.parent.mkdir(parents=True, exist_ok=True)       # 目录不存在就创建
    payload = {"timestamp": int(time.time()),              # 生成时间戳（秒）
               "model": config.LLM_MODEL,                  # 评估 LLM 名称
               "embedding": "bge-m3",                      # 评估 Embedding 名称
               "summary": result["summary"],               # 四个指标的均分
               "details": result["details"]}               # 逐条明细
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),   # 序列化，保留中文
                      encoding="utf-8")                    # 指定 UTF-8 编码
    logger.info("评测结果已保存：%s", target)                # 记录写入路径
    return target                                          # 返回写入路径


def main() -> None:
    """命令行入口：python -m eval_ragas。"""
    logger.info("开始 RAGAS 评测：评测集 %s，共 %d 条，评估模型 %s",
                EVAL_SET.name, len(load_eval_set()), config.LLM_MODEL)   # 打印启动信息
    result = run_evaluation()                              # 跑完整评测
    save_result(result)                                    # 结果落盘
    logger.info("评测结束，四个指标均分：%s", result["summary"])   # 打印最终汇总


if __name__ == "__main__":                                 # 支持 python -m eval_ragas
    main()                                                 # 执行命令行入口
