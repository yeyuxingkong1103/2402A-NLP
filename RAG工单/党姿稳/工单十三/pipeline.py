# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG项目-RAG 性能瓶颈识别与优化
# 模块职责：RAG 各阶段的共享实现（向量检索、BM25、RRF 融合、MMR 去冗余、上下文组装、生成、后处理）

import time                                                     # 导入 time 用于统计子步骤耗时
import numpy as np                                              # 导入 numpy 用于矩阵运算
import requests                                                 # 导入 requests 用于调用 LLM 接口
import config                                                   # 导入本工单配置模块

_EPS = 1e-9                                                     # 防止除零的极小量


# ---------------------------------------------------------------------------
# 一、阶段二「检索」：向量检索的两种实现（优化前/优化后）
# ---------------------------------------------------------------------------
def vector_search_naive(qvec, vecs, top_k):                     # 定义未优化的向量检索
    """每次查询都对全量语料向量重新计算范数并归一化，再与归一化查询做点积。"""
    qn = qvec / max(float(np.linalg.norm(qvec)), _EPS)          # 归一化查询向量
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)         # 每次查询重算 1950 个行范数（重复开销）
    unit = vecs / np.maximum(norms, _EPS)                       # 生成归一化副本（额外内存与拷贝）
    sims = unit.dot(qn)                                         # 余弦相似度 = 单位向量点积
    order = np.argsort(-sims)[:top_k]                           # 取相似度最高的候选行号
    return [(int(i), float(sims[i])) for i in order]            # 返回 (行号, 相似度) 列表


def vector_search_fast(qvec, unit_vecs, top_k):                 # 定义优化后的向量检索
    """直接使用启动时预归一化的连续矩阵做单次矩阵乘法，查询侧只归一化一次。"""
    qn = qvec / max(float(np.linalg.norm(qvec)), _EPS)          # 归一化查询向量
    sims = unit_vecs.dot(qn)                                    # 单次矩阵乘法得到全部相似度
    order = np.argpartition(-sims, top_k)[:top_k]               # 用 partition 取前 top_k（O(n) 而非全排序）
    order = order[np.argsort(-sims[order])]                     # 仅对这 top_k 个候选排序
    return [(int(i), float(sims[i])) for i in order]            # 返回 (行号, 相似度) 列表


def bm25_search(idx, query, top_k):                             # 定义 BM25 检索步骤
    """调用 BM25 索引检索，返回 (行号, 得分) 列表。"""
    return [(int(i), float(s)) for i, s in idx.search(query, top_k=top_k)]  # 统一转为原生类型


def expand_query(query, keywords):                              # 定义查询扩写函数
    """用阶段一的关键词扩写 BM25 检索文本；关键词已在原文中出现的不重复拼接。"""
    extra = [w for w in keywords if w and w not in query]       # 只保留原文中没有的新词
    return query if not extra else query + " " + " ".join(extra)  # 无新词时退回原查询


# ---------------------------------------------------------------------------
# 二、阶段二「检索」：RRF 融合与 MMR 去冗余
# ---------------------------------------------------------------------------
def rrf_fuse(rank_lists, k=None):                               # 定义 RRF 融合函数
    """倒数排名融合：多路召回按 1/(k+排名) 累加得分，返回 (行号, 得分) 降序列表。"""
    k = k or config.RRF_K                                       # 融合平滑常数
    scores = {}                                                 # 行号 -> 融合得分
    for rows in rank_lists:                                     # 遍历每一路召回
        for rank, row in enumerate(rows):                       # 遍历该路的名次
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank + 1)  # 累加倒数排名得分
    return sorted(scores.items(), key=lambda x: -x[1])          # 按得分降序返回


def mmr_dedup(cand, unit_vecs, top_k, lam=None):                # 定义 MMR 去冗余函数
    """最大边际相关：在相关性与多样性之间折中，剔除与已选块近乎重复的候选。"""
    lam = config.MMR_LAMBDA if lam is None else lam             # 相关性权重
    picked = []                                                 # 已选中的行号
    pool = [row for row, _ in cand]                             # 候选行号（保持融合得分顺序）
    sim_map = dict(cand)                                        # 行号 -> 融合得分
    while pool and len(picked) < top_k:                         # 选够或候选耗尽即停止
        best, best_val = None, -1e9                             # 本轮最优候选与得分
        for row in pool:                                        # 遍历剩余候选
            rel = sim_map[row]                                  # 相关性：RRF 融合得分
            if picked:                                          # 已有选中块时计算冗余度
                sims = unit_vecs[picked].dot(unit_vecs[row])    # 与已选块的余弦相似度
                red = float(np.max(sims))                       # 取最大相似度作为冗余度
            else:                                               # 首个块无冗余概念
                red = 0.0                                       # 冗余度为 0
            val = lam * rel - (1 - lam) * red                   # MMR 目标函数
            if val > best_val:                                  # 记录更优候选
                best, best_val = row, val                       # 更新最优
        picked.append(best)                                     # 选中该候选
        pool.remove(best)                                       # 从候选池移除
    return picked                                               # 返回最终选中的行号


# ---------------------------------------------------------------------------
# 三、阶段三「上下文组装与提示工程」
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = "你是招股说明书问答助手。只依据给定资料回答，并在答案中标注来源与页码。"  # 系统提示词


def assemble_context(chunks, rows, max_chars=None):             # 定义上下文组装函数
    """把选中的语料块拼成带来源标注的上下文文本，并裁剪到长度上限。"""
    max_chars = max_chars or config.CONTEXT_MAX_CHARS           # 上下文长度上限
    parts, used = [], 0                                         # 片段列表与已用字符数
    for row in rows:                                            # 按最终排序遍历选中块
        c = chunks[row]                                         # 取出语料块
        text = "【%s 第 %d 页】\n%s" % (c["source"], c["page"], c["text"])  # 带来源标注渲染
        if used + len(text) > max_chars:                        # 超出上限则停止拼接
            break                                               # 结束循环
        parts.append(text)                                      # 追加片段
        used += len(text)                                       # 累计字符数
    return "\n\n".join(parts)                                   # 用空行分隔拼接为上下文字符串


def build_prompt(context, query):                               # 定义提示词组装函数
    """把上下文与问题组装为发送给 LLM 的用户提示词。"""
    return ("资料：\n%s\n\n问题：%s\n\n请依据资料作答，分点列出，标注来源说明书与页码；"
            "资料未涵盖的内容请写「资料未涵盖」。" % (context, query))  # 返回完整提示词


# ---------------------------------------------------------------------------
# 四、阶段四「LLM 生成」与阶段五「后处理与响应格式化」
# ---------------------------------------------------------------------------
def call_llm(system, user, temperature=0.0, max_tokens=800):    # 定义 LLM 调用函数
    """调用 DeepSeek 兼容接口生成答案，返回 (答案文本, 生成 token 数)。"""
    url = config.DEEPSEEK_BASE_URL.rstrip("/") + "/chat/completions"  # 拼装接口地址
    payload = {"model": config.DEEPSEEK_MODEL,                  # 模型名称
               "messages": [{"role": "system", "content": system},  # 系统提示
                            {"role": "user", "content": user}],  # 用户提示
               "temperature": temperature,                      # 采样温度（0 表示确定性输出）
               "max_tokens": max_tokens}                        # 最大生成 token 数
    headers = {"Authorization": "Bearer " + config.DEEPSEEK_API_KEY,  # 鉴权头
               "Content-Type": "application/json"}              # 请求体类型
    resp = requests.post(url, json=payload, headers=headers, timeout=config.LLM_TIMEOUT)  # 发送请求
    resp.raise_for_status()                                     # 非 2xx 状态抛出异常
    data = resp.json()                                          # 解析响应
    text = data["choices"][0]["message"]["content"]             # 取答案正文
    tokens = (data.get("usage") or {}).get("completion_tokens", 0)  # 取生成 token 数
    return text.strip(), int(tokens)                            # 返回答案与 token 数


def format_response(answer, chunks, rows):                      # 定义后处理格式化函数
    """整理引用列表并把答案与引用拼装为最终响应结构。"""
    cites, seen = [], set()                                     # 引用列表与去重集合
    for rank, row in enumerate(rows, 1):                        # 遍历最终选中的块
        c = chunks[row]                                         # 取出语料块
        key = (c["source"], c["page"])                          # 引用唯一键
        if key in seen:                                         # 已记录则跳过
            continue                                            # 继续下一个
        seen.add(key)                                           # 标记已记录
        cites.append("[%d] %s 第 %d 页" % (rank, c["source"], c["page"]))  # 渲染引用条目
    return {"answer": answer, "citations": cites}               # 返回结构化响应


def elapsed(t0):                                                # 定义耗时计算辅助函数
    """返回自 t0 起经过的秒数，统一保留六位小数。"""
    return round(time.perf_counter() - t0, 6)                   # 高精度差值


# ---------------------------------------------------------------------------
# 五、阶段三~五的共用流程（优化前后两版完全一致，只有查询理解与检索实现不同）
# ---------------------------------------------------------------------------
def run_tail(query, chunks, rows, tracer):                      # 定义后三阶段共用流程
    """执行上下文组装、LLM 生成与后处理，并写入阶段追踪记录。"""
    tracer.begin(config.STAGE_CONTEXT)                          # 阶段三开始
    context = assemble_context(chunks, rows)                    # 组装上下文
    prompt = build_prompt(context, query)                       # 组装提示词
    tracer.end(config.STAGE_CONTEXT, context_chars=len(context), prompt_chars=len(prompt))  # 阶段三结束
    tracer.begin(config.STAGE_GENERATE)                         # 阶段四开始
    answer, tokens = call_llm(_SYSTEM_PROMPT, prompt)           # 调用 LLM 生成答案
    tracer.end(config.STAGE_GENERATE, completion_tokens=tokens)  # 阶段四结束
    tracer.begin(config.STAGE_POST)                             # 阶段五开始
    response = format_response(answer, chunks, rows)            # 整理引用与响应结构
    tracer.end(config.STAGE_POST, citation_count=len(response["citations"]))  # 阶段五结束
    return response                                             # 返回结构化响应
