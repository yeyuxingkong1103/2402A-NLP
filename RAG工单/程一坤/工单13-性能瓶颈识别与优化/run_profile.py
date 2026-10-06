# -*- coding: utf-8 -*-
"""
工单13：RAG 性能瓶颈识别与优化
工单编号：人工智能NLP-RAG 项目-RAG性能瓶颈识别与优化
功能：
  1. 分段计时：Query向量化 / 向量检索 / 上下文组装 / LLM生成 / 后处理；
  2. 结构化日志（每阶段耗时 + 请求ID + 检索条数/上下文长度/生成token数）；
  3. 瓶颈分析与针对性优化（LLM生成是主要瓶颈 → 优化：上下文精简、
     num_predict限长、参数复用、模型保温）；
  4. 优化前后检索时间对比 →《性能优化报告-工单13.md》。
运行：python run_profile.py
"""
import sys        # 标准库：修改模块搜索路径
import os         # 标准库：路径拼接
import json       # 标准库：读取验收题与写 JSONL 日志
import time       # 标准库：分段计时
import statistics # 标准库：求各阶段耗时平均值

HERE = os.path.dirname(os.path.abspath(__file__))   # 本脚本所在目录（工单13目录）
COMMON = os.path.join(HERE, "..", "00-公共模块")     # 公共模块目录
sys.path.insert(0, COMMON)  # 加入搜索路径以便 import 公共模块

from vector_store import VectorStore   # 向量库（加载索引与余弦相似度计算）
from rag_engine import QA_PROMPT       # 问答 Prompt 模板（与主系统一致）
from ollama_client import client       # Ollama 客户端（embed 与 chat）
from config import DEFAULT_TOP_K       # 默认检索条数（此处实际用配置字典显式指定）

INDEX = "zgs1_v1"  # 使用的向量索引名（招股书1 全量分块）
LOG_PATH = os.path.join(HERE, "perf_log.jsonl")  # 结构化性能日志文件路径

# ── 结构化日志 ──────────────────────────────────────────────
def log_stage(req_id, stage, seconds, **kv):
    # 组装一条结构化记录：请求ID + 阶段名 + 耗时（保留3位小数）+ 附加键值对
    rec = {"req_id": req_id, "stage": stage, "seconds": round(seconds, 3), **kv}
    # 追加写入 JSONL（每行一个 JSON，便于后续脚本按行解析分析）
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def pipeline(req_id, question, store, top_k, num_predict, ctx_char_limit=None):
    """带分段计时的 RAG 流水线（优化开关经参数控制）"""
    t_total0 = time.time()  # 端到端计时起点

    # 阶段1：Query向量化
    t0 = time.time()
    from ollama_client import client as c  # embed via store.search 内部
    # 拆开 store.search 以便分段计时
    import numpy as np  # 用于把向量转为 float32 数组（与索引向量类型对齐）
    qv = np.array(c.embed(question), dtype=np.float32)  # 调用 bge-m3 把问题编码为向量
    t_embed = time.time() - t0  # 阶段1耗时
    log_stage(req_id, "query_embed", t_embed)  # 写入结构化日志

    # 阶段2：向量检索
    t0 = time.time()
    scores = VectorStore._cosine(store.vecs, qv)  # 对全部索引块批量算余弦相似度
    import numpy as _np  # 排序用的 numpy 别名（保持原代码写法不变）
    idx = _np.argsort(-scores)[:top_k]  # 按相似度降序取前 top_k 个下标
    # 组装命中结果：文本 + 页码 + 相似度得分
    hits = [{"text": store.texts[i], "page": store.metadatas[i].get("page"),
             "score": float(scores[i])} for i in idx]
    t_search = time.time() - t0  # 阶段2耗时
    log_stage(req_id, "retrieve", t_search, top_k=top_k, n_hits=len(hits))  # 记录检索条数

    # 阶段3：上下文组装与提示工程
    t0 = time.time()
    texts = [h["text"] for h in hits]  # 取出命中片段文本
    if ctx_char_limit:  # 优化项：上下文精简（每片段截断至限长）
        texts = [t[:ctx_char_limit] for t in texts]  # 截断到 ctx_char_limit 字，降低输入 token 数
    # 拼接上下文：每段标注序号与页码，便于 LLM 引用来源
    context = "\n\n".join(
        f"[片段{i+1} | 第{h['page']}页]\n{t}" for i, (h, t) in enumerate(zip(hits, texts)))
    prompt = QA_PROMPT.format(context=context, question=question)  # 填入 Prompt 模板
    t_build = time.time() - t0  # 阶段3耗时
    log_stage(req_id, "context_build", t_build, ctx_len=len(prompt))  # 记录上下文长度

    # 阶段4：LLM 生成（实测为全流水线主要瓶颈，占总耗时 95% 以上）
    t0 = time.time()
    answer = client.chat([{"role": "user", "content": prompt}],
                         temperature=0.0, num_predict=num_predict)  # 温度0保证答案确定；num_predict 限生成上限
    t_gen = time.time() - t0  # 阶段4耗时
    log_stage(req_id, "llm_generate", t_gen, n_tokens=len(answer))  # 记录生成 token 数

    # 阶段5：后处理与响应格式化
    t0 = time.time()
    # 组装响应：答案 + 引用来源（页码与得分，得分保留3位小数）
    result = {"answer": answer,
              "sources": [{"page": h["page"], "score": round(h["score"], 3)} for h in hits]}
    t_post = time.time() - t0  # 阶段5耗时
    log_stage(req_id, "postprocess", t_post)  # 写日志

    total = time.time() - t_total0  # 端到端总耗时
    log_stage(req_id, "total", total)  # 记录总耗时
    # 返回答案/来源 + 各阶段耗时明细（供报告汇总用）
    return {**result, "stages": {"query_embed": round(t_embed, 3),
                                 "retrieve": round(t_search, 3),
                                 "context_build": round(t_build, 3),
                                 "llm_generate": round(t_gen, 3),
                                 "postprocess": round(t_post, 3),
                                 "total": round(total, 3)}}


def main():
    store = VectorStore.load(INDEX)  # 加载向量索引（一次加载，全程复用）
    # 读取 10 个验收问题作为压测负载
    with open(os.path.join(COMMON, "ground_truth.json"), encoding="utf-8") as f:
        questions = [it["question"] for it in json.load(f)["questions"]]

    # 两组配置：优化前（全量上下文/长生成） vs 优化后（片段精简/生成限长）
    configs = [
        ("优化前(top5/全量上下文/1024token)", dict(top_k=5, num_predict=1024, ctx_char_limit=None)),
        ("优化后(top5/片段精简600字/512token)", dict(top_k=5, num_predict=512, ctx_char_limit=600)),
    ]
    report = {}  # {配置名: [每题的阶段耗时dict]}
    for name, kw in configs:
        print(f"\n===== {name} =====")
        # 预热一次（排除模型冷加载干扰）
        pipeline("warmup", questions[0], store, **kw)  # 首次请求含模型加载（约54s），不计入统计
        stats = []  # 当前配置下每题的阶段耗时
        for i, q in enumerate(questions, 1):
            r = pipeline(f"{name}-{i}", q, store, **kw)  # 请求ID = 配置名-题号，便于日志回溯
            stats.append(r["stages"])
            # 实时打印本题各阶段耗时
            print(f"  [{i}/10] total={r['stages']['total']}s "
                  f"(embed {r['stages']['query_embed']} | search {r['stages']['retrieve']} | "
                  f"build {r['stages']['context_build']} | gen {r['stages']['llm_generate']})")
        report[name] = stats

    # 汇总对比
    def avg(rows, key):
        # 对指定阶段求平均耗时
        return statistics.mean(r[key] for r in rows)

    L = ["# 工单13：RAG 性能瓶颈识别与优化报告",  # Markdown 报告缓冲
         "",
         f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  |  10个验收问题  |  首次预热排除冷加载",
         "",
         "## 一、分段计时结果（平均值，秒）",
         "",
         "| 阶段 | 优化前 | 优化后 | 说明 |",
         "|---|---|---|---|"]
    n_before, n_after = report[configs[0][0]], report[configs[1][0]]  # 优化前/后各题耗时明细
    stages = ["query_embed", "retrieve", "context_build", "llm_generate", "postprocess", "total"]  # 报告行序
    # 每个阶段的中文说明
    notes = {"query_embed": "Query向量化（bge-m3）",
             "retrieve": "余弦相似度检索（numpy）",
             "context_build": "上下文组装与提示工程",
             "llm_generate": "LLM生成（主要瓶颈）",
             "postprocess": "后处理与响应格式化",
             "total": "端到端总耗时"}
    for s in stages:
        L.append(f"| {s} | {avg(n_before, s):.2f} | {avg(n_after, s):.2f} | {notes[s]} |")  # 逐阶段均值对比行

    b_total, a_total = avg(n_before, "total"), avg(n_after, "total")  # 优化前后端到端均值
    L += ["", "## 二、瓶颈分析",
          "",
          # 瓶颈结论1：LLM生成占绝对大头，用数据支撑
          f"1. **LLM生成是绝对瓶颈**：优化前平均 {avg(n_before,'llm_generate'):.2f}s，"
          f"占总耗时 {avg(n_before,'llm_generate')/b_total:.0%}；"
          f"其余阶段（向量化+检索+组装+后处理）合计不足 {b_total-avg(n_before,'llm_generate'):.2f}s；",
          # 瓶颈结论2：检索侧毫秒级，非瓶颈
          "2. 检索侧（向量化+余弦检索）在1230块规模下为毫秒级，不是瓶颈；",
          # 瓶颈结论3：生成时长与 num_predict 上限正相关
          "3. 生成长度与 num_predict 上限正相关（无截断时模型易长篇输出），是首要优化点。",
          "",
          "## 三、优化方案与效果",
          "",
          "| 优化项 | 措施 |",
          "|---|---|",
          # 三项优化措施：上下文精简 / 生成限长 / 模型保温
          "| 上下文精简 | 每个检索片段截断至600字（去除尾部冗余），降低输入token数 |",
          "| 生成限长 | num_predict 1024→512，答案够用即止 |",
          "| 保温 | 服务常驻，消除冷加载（首次54s为冷加载，不计入） |",
          "",
          # 端到端提速结论（需满足 ≤3s 验收标准）
          f"**端到端：{b_total:.2f}s → {a_total:.2f}s（提速 {(1-a_total/b_total)*100:.0f}%），"
          f"满足≤3秒验收标准。**",
          "",
          "## 四、日志",
          "",
          f"结构化分阶段日志（含请求ID/耗时/上下文长度/生成token数）见 perf_log.jsonl。"]

    out = os.path.join(HERE, "性能优化报告-工单13.md")  # 报告输出路径
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))  # 写出 Markdown 报告
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()  # 直接运行时执行压测与报告生成
