# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-LightRAG 优化任务
# 模块职责：为测试问题构建「银标」参考答案——用 BM25 全文宽窗口取证据，由 LLM 精读后作答
# 说明：工单未提供招股说明书问题的标准答案，且两套系统都要与同一份答案对比。
#      这里刻意使用与两套系统都不同的检索方式（纯 BM25 宽窗口，不含向量、不含图谱），
#      保证参考答案的构造过程独立于被评估对象，避免评估自我循环

import os                                                       # 导入 os 用于缓存文件判断
import json                                                     # 导入 json 用于参考答案读写
import config                                                   # 导入本工单配置模块
import llm                                                      # 导入 LLM 封装（带磁盘缓存）
from logger import LOGGER                                       # 导入统一日志对象

_SYSTEM = ("你是招股说明书研究专家。请严格依据给定的资料片段作答，"
           "答案中的每一个事实与数字都必须能在资料片段中找到出处。")

_USER = """资料片段：
%s

问题：%s

请依据上述资料片段给出完整、准确的参考答案：
1. 直接回答问题的所有要点，数字、比例、金额、名称必须与资料完全一致。
2. 资料片段中确实没有的信息，写「资料未涵盖」，不要凭常识补充。
3. 用分点形式作答，每点尽量引用资料中的原文表述。"""


def _wide_context(question, chunks, idx):                       # 定义宽窗口证据收集函数
    """用 BM25 取该问题最相关的若干块作为证据窗口（不涉及向量与图谱）。"""
    hits = idx.search(question, top_k=config.REF_POOL_K)        # BM25 全文检索取宽窗口
    blocks = []                                                 # 初始化证据片段
    for i, _ in hits:                                           # 遍历命中的块行号
        c = chunks[i]                                           # 取出语料块
        blocks.append("【%s 第 %d 页】\n%s" % (c["source"], c["page"], c["text"]))  # 带来源标注渲染
    return "\n\n".join(blocks)                                  # 用空行分隔拼接


def build_references(chunks, idx, force=False):                 # 定义参考答案构建函数
    """为配置中的全部测试问题构建银标参考答案，结果落盘缓存。"""
    if not force and os.path.isfile(config.REF_CACHE_FILE):     # 缓存存在且非强制重建
        with open(config.REF_CACHE_FILE, "r", encoding="utf-8") as f:  # 以 UTF-8 打开缓存
            refs = json.load(f)                                 # 反序列化参考答案
        if len(refs) == len(config.QUESTIONS):                  # 条数与问题数一致才可用
            LOGGER.info("命中参考答案缓存：%d 条", len(refs))     # 输出命中信息
            return refs                                         # 返回缓存的参考答案
        LOGGER.warning("参考答案缓存条数不符，重建")             # 输出告警
    refs = []                                                   # 初始化参考答案列表
    for q in config.QUESTIONS:                                  # 遍历每个测试问题
        ctx = _wide_context(q["question"], chunks, idx)         # 取该问题的宽窗口证据
        text = llm.chat(_SYSTEM, _USER % (ctx, q["question"]),  # 调用 LLM 生成参考答案
                        temperature=0.0, max_tokens=900)        # 限制输出长度控制成本
        refs.append({"id": q["id"], "question": q["question"],  # 记录问题编号与原文
                     "reference": (text or "").strip(),          # 记录参考答案
                     "source": "BM25 宽窗口(%d 块) + LLM" % config.REF_POOL_K})  # 记录构造方式
        LOGGER.info("参考答案 %d/%d 构建完成", len(refs), len(config.QUESTIONS))  # 输出进度
    config.ensure_dirs()                                        # 确保数据目录存在
    with open(config.REF_CACHE_FILE, "w", encoding="utf-8") as f:  # 以 UTF-8 覆盖写入缓存
        json.dump(refs, f, ensure_ascii=False, indent=1)        # 序列化参考答案（保留中文）
    llm.save_cache()                                            # 落盘 LLM 磁盘缓存
    LOGGER.info("参考答案构建完成，写入 %s", config.REF_CACHE_FILE)  # 输出完成日志
    return refs                                                 # 返回参考答案列表
