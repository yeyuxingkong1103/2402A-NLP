# -*- coding: utf-8 -*-
"""
工单02：检索准确率优化（优化前 vs 优化后 对比实验）
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
优化项：
  ① 页眉页脚清洗（降噪，防止噪音稀释向量语义）
  ② LLM 查询扩展 + 多路召回 RRF 融合（解决问法与文档用词不一致）
  ③ 关键词重排（字面证据加分）
运行：python run_optimize.py  →  生成《优化前后对比结果.md》
"""
import sys
import os
import json
import time

# 当前脚本所在目录（用于定位 ground_truth.json 与输出报告路径）
HERE = os.path.dirname(os.path.abspath(__file__))
# 公共模块目录路径（config/pdf_parser/vector_store 等都在这里）
COMMON = os.path.join(HERE, "..", "00-公共模块")
# 把公共模块目录加入模块搜索路径
sys.path.insert(0, COMMON)

# PDF 路径配置
from config import PDF_ZGS1
# PDF 解析（clean_headers 参数可控制是否清洗页眉，用于构建基线索引）
from pdf_parser import parse_pdf_text
# 文本分块
from chunker import chunk_pages
# 自研向量库（numpy + JSON 持久化）
from vector_store import VectorStore
# RAG 引擎（baseline_ask 用其 ask 单路检索）与 QA_PROMPT 提示词模板
from rag_engine import RAGEngine, QA_PROMPT
# 工单02核心优化组件：查询扩展 / 多路召回RRF融合 / 关键词重排
from query_optimize import expand_query, multi_retrieve, keyword_rerank
# Ollama 客户端：optimized_ask 中直接调用 LLM 生成
from ollama_client import client

BEFORE_INDEX = "zgs1_before"  # 优化前基线索引（不清洗页眉）
AFTER_INDEX = "zgs1_v1"       # 工单01 优化后索引（清洗页眉）
TOP_K = 5  # 两侧统一取 Top5 候选，保证对比公平


def build_before_index():
    """构建优化前基线索引：不清洗页眉页脚（还原工单01优化前的状态）"""
    # 基线索引已存在则直接返回（幂等，避免重复向量化耗时）
    if VectorStore.load(BEFORE_INDEX) is not None:
        return
    print("构建优化前基线索引（不清洗页眉）...")
    # 关键差异：clean_headers=False 保留页眉噪音行，还原"未优化"的检索质量
    pages = parse_pdf_text(PDF_ZGS1, clean_headers=False)
    chunks = chunk_pages(pages)
    store = VectorStore()
    store.add(chunks, source_name="招股说明书1.pdf")
    # 持久化为 zgs1_before，与工单01的 zgs1_v1 互不影响
    store.save(BEFORE_INDEX)


def normalize(s):
    """归一化：去空格/逗号，便于数字事实匹配"""
    # 去掉空格/逗号/换行后比较，避免"1,000万"与"1000万"这种格式差异导致误判未命中
    return str(s).replace(" ", "").replace(",", "").replace("\n", "")


def fact_hit_ratio(key_facts, text):
    """关键事实命中率：命中数/事实总数"""
    if not key_facts:
        # 该题无关键事实定义时返回 -1，与 0%~100% 区分开，不计入统计
        return -1
    t = normalize(text)
    # 每个关键事实归一化后在文本中做子串匹配，命中数除以事实总数得比率
    return sum(1 for f in key_facts if normalize(f) in t) / len(key_facts)


def baseline_ask(engine, question):
    """优化前流水线：单路向量检索 Top5 → LLM"""
    # 直接用工单01的 RAGEngine.ask：用户原句单路向量召回，无扩展、无重排
    return engine.ask(question, top_k=TOP_K, with_context=True)


def optimized_ask(store, question):
    """优化后流水线：查询扩展 → 多路召回RRF → 关键词重排 → LLM"""
    t0 = time.time()
    # 优化②-1：让 LLM 把原问题改写出 3 个等价问法（弥合问法与文档用词差异）
    variants = expand_query(question, n=3)
    # 优化②-2：每个问法各自向量召回，RRF 倒数排名融合，降低单路召回的偶然性
    hits = multi_retrieve(store, variants, top_k=TOP_K)
    # 优化③：对融合结果按 jieba 关键词命中加权重排，补上字面证据校验
    hits = keyword_rerank(question, hits)
    retrieval_time = time.time() - t0

    # 把 Top5 分块拼装成带页码标注的上下文（与工单01的上下文格式保持一致）
    context = "\n\n".join(
        f"[片段{i+1} | 第{h['page']}页]\n{h['text']}" for i, h in enumerate(hits))
    # 用公共 QA_PROMPT 模板填充上下文与问题
    prompt = QA_PROMPT.format(context=context, question=question)
    # 生成侧护栏：历史沿革中会出现旧值，要求以现行/最新数值作答
    prompt += "\n注意：若参考上下文中同一指标出现多个不同数值（如公司历史沿革中的历史注册资本），以最新、现行的数值为准作答。"
    t1 = time.time()
    # 调用本地 qwen2.5:7b-instruct 生成；temperature=0.1 降低随机性保证事实稳定
    answer = client.chat([{"role": "user", "content": prompt}],
                         temperature=0.1, num_predict=600)
    gen_time = time.time() - t1
    # 返回与 RAGEngine.ask 兼容的结果结构，便于统一计算命中率与耗时
    return {
        "question": question, "answer": answer,
        "sources": [{"page": h["page"]} for h in hits],
        "context": "\n".join(h["text"] for h in hits),
        "retrieval_time": round(retrieval_time, 2),
        "generation_time": round(gen_time, 2),
        "time_cost": round(retrieval_time + gen_time, 2),
        "variants": variants,  # 保留改写问法，报告中展示扩展效果
    }


def main():
    # 读取评测集：每题含 question、key_facts（关键事实列表）、answer_hint（标准答案要点）
    with open(os.path.join(COMMON, "ground_truth.json"), encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    # 构建优化前基线索引（若不存在）
    build_before_index()
    # 优化前侧：基于基线索引的 RAGEngine
    before_engine = RAGEngine(BEFORE_INDEX)
    # 优化后侧：直接加载工单01清洗过页眉的索引 zgs1_v1
    after_store = VectorStore.load(AFTER_INDEX)

    rows = []
    # 四个计数器：优化前/后 × 检索命中 / 回答正确，用于汇总指标表
    n_before_ok = n_after_ok = n_before_hit = n_after_hit = 0
    for i, item in enumerate(questions, 1):
        q, facts = item["question"], item["key_facts"]
        print(f"[{i}/{len(questions)}] {q}")

        # 优化前：单路检索问答
        rb = baseline_ask(before_engine, q)
        # 检索命中：判断关键事实是否出现在召回的 Top5 上下文中
        # rb["retrieved"] and ... or "" 是兜底写法：retrieved 为空列表时整个表达式取 ""
        b_hit = fact_hit_ratio(facts, rb["retrieved"] and "\n".join(h["text"] for h in rb["retrieved"]) or "")
        # 回答正确：判断关键事实是否出现在 LLM 最终答案里
        b_ans = fact_hit_ratio(facts, rb["answer"])

        # 优化后：扩展+多路召回+重排的问答
        ra = optimized_ask(after_store, q)
        a_hit = fact_hit_ratio(facts, ra["context"])
        a_ans = fact_hit_ratio(facts, ra["answer"])

        # 判定阈值 0.5：命中过半关键事实即记为"命中/正确"（半对也给分，宽松可复现）
        n_before_hit += b_hit >= 0.5; n_after_hit += a_hit >= 0.5
        n_before_ok += b_ans >= 0.5;  n_after_ok += a_ans >= 0.5
        rows.append({"item": item, "b_hit": b_hit, "b_ans": b_ans, "rb": rb,
                     "a_hit": a_hit, "a_ans": a_ans, "ra": ra})

    total = len(questions)
    # ── 生成对比报告 ──
    # 报告第一、二部分：优化方案表 + 总体指标表（命中率/正确率前后对比与提升幅度）
    L = [
        "# 工单02：检索优化前后对比报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  |  LLM：qwen2.5:7b-instruct  |  Embedding：bge-m3",
        "",
        "## 一、优化方案",
        "",
        "| 优化项 | 优化前（工单01基础版） | 优化后 | 针对的问题 |",
        "|---|---|---|---|",
        "| 页眉页脚清洗 | 保留噪音行 | 正则清洗公司名/页码行 | 噪音稀释向量语义，真值分块排名被压低 |",
        "| 查询扩展 | 用户原句单路检索 | LLM改写3个等价问法 | 问法与文档用词不一致（如问'军用收入'文档写'军品收入'）|",
        "| 多路召回融合 | 单路Top5 | 多路召回+RRF倒数排名融合 | 单次召回偶然性大 |",
        "| 关键词重排 | 纯向量分 | jieba关键词命中加权 | 向量语义漂移，缺字面证据校验 |",
        "",
        "## 二、总体指标",
        "",
        "| 指标 | 优化前 | 优化后 | 提升 |",
        "|---|---|---|---|",
        # :.0% 百分比格式化；提升列 = (优化后-优化前)/总数
        f"| 检索命中率（Top5上下文含关键事实） | {n_before_hit}/{total} = {n_before_hit/total:.0%} | {n_after_hit}/{total} = {n_after_hit/total:.0%} | +{(n_after_hit-n_before_hit)/total:.0%} |",
        f"| 回答正确率（答案含关键事实） | {n_before_ok}/{total} = {n_before_ok/total:.0%} | {n_after_ok}/{total} = {n_after_ok/total:.0%} | +{(n_after_ok-n_before_ok)/total:.0%} |",
        "",
        "## 三、逐题对比",
        "",
        "| ID | 问题 | 优化前检索命中 | 优化前回答正确 | 优化后检索命中 | 优化后回答正确 | 优化后来源页 |",
        "|---|---|---|---|---|---|---|",
    ]
    # 逐题表格行：问题截断到22字防表格过宽；✅/❌ 按 0.5 阈值判定；来源页只列前3个
    for r in rows:
        q = r["item"]["question"]
        L.append(f"| {r['item']['id']} | {q[:22]}… | {r['b_hit']:.0%} | {'✅' if r['b_ans']>=0.5 else '❌'} "
                 f"| {r['a_hit']:.0%} | {'✅' if r['a_ans']>=0.5 else '❌'} "
                 f"| {','.join(str(s['page']) for s in r['ra']['sources'][:3])} |")
    L += ["", "## 四、逐题明细", ""]
    # 逐题明细：优化前后回答各截取前200字；同时展示 LLM 改写的问法（variants[1:] 是扩展问法）
    for r in rows:
        L += [
            f"### 问题 {r['item']['id']}",
            f"**Q：** {r['item']['question']}",
            f"**标准答案要点：** {r['item']['answer_hint']}",
            "",
            f"**优化前回答**（命中 {r['b_hit']:.0%}）：{r['rb']['answer'][:200]}",
            "",
            f"**优化后回答**（命中 {r['a_hit']:.0%}，改写问法：{' / '.join(r['ra']['variants'][1:]) or '无'}）：{r['ra']['answer'][:200]}",
            "",
            "---", "",
        ]
    # 输出报告到工单02目录，UTF-8 编码
    out = os.path.join(HERE, "优化前后对比结果.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    # 终端同步打印汇总结论，便于运行结束快速核对
    print(f"\n总体: 检索命中 {n_before_hit}/{total} → {n_after_hit}/{total} | "
          f"回答正确 {n_before_ok}/{total} → {n_after_ok}/{total}")
    print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
