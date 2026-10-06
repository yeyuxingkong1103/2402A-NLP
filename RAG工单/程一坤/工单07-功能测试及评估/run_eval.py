# -*- coding: utf-8 -*-
"""
工单07：功能测试及评估
工单编号：人工智能NLP-RAG-功能测试及评估
功能：用工单01-06 实现的 RAG 系统对 ccf_competition 9 份金融年报进行测试：
  10 个问题 → RAG检索结果 + 生成回答 + 三维评估（检索命中率/回答正确率/LLM忠实度评分），
  输出《测试及评估报告-工单07.md》，并分析检索结果存在的问题。
运行：python run_eval.py
"""
import sys
import os
import json
import time

# 当前脚本目录与公共模块路径，保证能 import 00-公共模块
HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "00-公共模块")
sys.path.insert(0, COMMON)

from vector_store import VectorStore  # 自研 numpy 向量库
from hybrid_retriever import BM25Index, hybrid_search, apply_rerank  # BM25与混合检索
from rag_engine import QA_PROMPT  # 统一问答提示词模板
from ollama_client import client  # Ollama 本地模型客户端

INDEX = "ccf_v1"  # build_ccf_index.py 建的金融年报索引
TOP_K = 6  # 最终送入上下文的片段数

# LLM judge 提示词：输出一行JSON（忠实度/相关性1-5分+短理由）
JUDGE_PROMPT = """你是RAG回答质量评估器。请根据【参考上下文】评估【回答】的质量，只输出一行JSON：
{"faithfulness": 1-5整数, "relevance": 1-5整数, "reason": "10字内理由"}
faithfulness=回答是否忠实于上下文（无编造），relevance=回答与问题的相关性。

【问题】{question}

【参考上下文】
{context}

【回答】
{answer}

【JSON】"""


def ask_rag(store, bm25, question):
    """复用工单01-06的RAG流水线（双轨：扩展向量轨 + 混合轨）"""
    # 局部导入避免顶层循环依赖
    from query_optimize import expand_query, multi_retrieve
    t0 = time.time()  # 计时起点
    # A轨：查询扩展生成3个变体，向量库多路召回
    variants = expand_query(question, n=3)
    a_hits = multi_retrieve(store, variants, top_k=4, fetch_k=10)
    # B轨：混合检索（向量0.4+全文0.6，金融数字更依赖字面匹配故全文权重更高）
    b_hits = hybrid_search(store, bm25, question, mode="hybrid",
                           vec_w=0.4, ft_w=0.6, top_k=4, fetch_k=30)
    seen, hits = set(), []  # seen 去重、hits 保序合并
    for h in a_hits + b_hits:
        if h["text"] not in seen:  # 两轨命中可能有重叠，按文本去重
            seen.add(h["text"]); hits.append(h)
    hits = hits[:TOP_K]  # 截取前6条作为最终上下文
    # 拼装上下文：每片段带序号与来源（年报txt无页码，只标来源公司年报）
    context = "\n\n".join(
        f"[片段{i+1} | {h['source']}]\n{h['text']}" for i, h in enumerate(hits))
    # temperature=0.0 保证评估场景回答确定性，500token 足够事实型回答
    answer = client.chat([{"role": "user",
                           "content": QA_PROMPT.format(context=context, question=question)}],
                         temperature=0.0, num_predict=500)
    # 返回答案、命中、去重后的来源列表、上下文原文（judge要用）与耗时
    return {"answer": answer, "hits": hits,
            "sources": sorted({h["source"] for h in hits}),
            "context": context, "time": round(time.time() - t0, 2)}


def normalize(s):
    # 评估前归一化：去空格/逗号/换行，避免格式差异导致事实匹配失败
    return str(s).replace(" ", "").replace(",", "").replace("\n", "")


def fact_ratio(facts, text):
    # 关键事实命中率 = 在归一化文本中命中的事实数 / 总数；无事实时返回 -1
    t = normalize(text)
    return sum(1 for f in facts if normalize(f) in t) / len(facts) if facts else -1


def llm_judge(question, context, answer):
    """LLM 评估：忠实度 + 相关性（RAG评估思想的轻量实现）"""
    try:
        # 注意：模板含JSON花括号，不能用 str.format（会KeyError），改用 replace
        # JUDGE_PROMPT 里的 {"faithfulness":...} 会被 format 当占位符解析，踩过 KeyError 坑
        prompt = (JUDGE_PROMPT
                  .replace("{question}", question)  # 填入问题
                  .replace("{context}", context[:3000])  # 上下文截断3000字防超模型窗口
                  .replace("{answer}", answer))  # 填入待评回答
        # temperature=0.0 + num_predict=60：judge 只需输出一行短JSON
        raw = client.generate(prompt, temperature=0.0, num_predict=60)
        import re
        # 用正则抠出第一个 {...} JSON块（模型可能输出多余说明文字）
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0))  # 解析为字典
        # 取忠实度/相关性（int转换防模型输出字符串数字）与理由，缺省兜底0
        return int(d.get("faithfulness", 0)), int(d.get("relevance", 0)), d.get("reason", "")
    except Exception as e:
        # judge 失败（JSON解析错/模型超时）返回 -1 标记，不中断整体评估
        return -1, -1, str(e)[:30]


def main():
    # 加载金融年报索引；不存在则提示先建索引
    store = VectorStore.load(INDEX)
    if store is None:
        print("[错误] 请先运行 build_ccf_index.py")
        return
    # 加载 BM25 索引，不存在就现场构建并持久化
    bm25 = BM25Index.load(INDEX)
    if bm25 is None:
        bm25 = BM25Index().build(store.texts, store.metadatas)
        bm25.save(INDEX)

    # 读取10个测试问题（含关键事实、答案提示、所属年报等字段）
    with open(os.path.join(HERE, "test_questions.json"), encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    results = []  # 逐题完整结果（原题字段+RAG输出+各评估指标）
    for i, item in enumerate(questions, 1):
        q, facts = item["question"], item["key_facts"]
        print(f"[{i}/{len(questions)}] {q}")
        # 跑双轨检索+生成
        r = ask_rag(store, bm25, q)
        # 检索命中率：关键事实是否出现在Top6上下文中
        c_ratio = fact_ratio(facts, r["context"])
        # 回答命中率：关键事实是否出现在生成答案中
        a_ratio = fact_ratio(facts, r["answer"])
        # LLM judge 打分（忠实度/相关性）+理由
        faith, rel, reason = llm_judge(q, r["context"], r["answer"])
        # 合并原题字段与评估结果
        results.append({**item, "rag": r, "c_ratio": c_ratio, "a_ratio": a_ratio,
                        "faith": faith, "rel": rel, "reason": reason})
        # 控制台实时打印每题摘要
        print(f"  A: {r['answer'][:80]}... | 检索命中{c_ratio:.0%} 回答命中{a_ratio:.0%} "
              f"忠实度{faith} 相关性{rel} | {r['time']}s")

    n = len(results)
    # 总体统计：检索命中（≥50%）、回答正确（≥50%）
    hit_n = sum(1 for r in results if r["c_ratio"] >= 0.5)
    acc_n = sum(1 for r in results if r["a_ratio"] >= 0.5)
    # judge 有效分数列表（-1表示judge失败，剔除后求均分）
    faiths = [r["faith"] for r in results if r["faith"] > 0]
    rels = [r["rel"] for r in results if r["rel"] > 0]

    # 报告头部 + 测试设置 + 总体结果表
    L = ["# 工单07：功能测试及评估报告",
         "",
         f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  |  知识库：ccf_competition 9份金融年报"
         f"（{len(store.texts)}个分块）",
         "",
         "## 一、测试设置",
         "",
         f"- 测试系统：工单01-06 实现的 RAG（双轨检索：查询扩展向量轨 + 混合检索轨）",
         f"- 测试问题：10 个（覆盖平安银行/中国平安/招商银行/邮储银行/中信证券/中国太保/招商证券/华泰证券年报，"
         f"题型含单事实抽取、多事实对比、开放归纳）",
         "- 评估维度：检索命中率（Top6上下文含关键事实比例≥50%）、回答正确率（答案含关键事实）、"
         "LLM评估（忠实度/相关性 1-5分）",
         "",
         "## 二、总体结果",
         "",
         f"| 指标 | 结果 |",
         f"|---|---|",
         f"| 检索命中率 | {hit_n}/{n} = {hit_n/n:.0%} |",
         f"| 回答正确率 | {acc_n}/{n} = {acc_n/n:.0%} |",
         f"| 忠实度(LLM评估均分) | {sum(faiths)/len(faiths):.1f}/5 |",
         f"| 相关性(LLM评估均分) | {sum(rels)/len(rels):.1f}/5 |",
         f"| 平均响应时间 | {sum(r['rag']['time'] for r in results)/n:.1f}s |",
         "",
         "## 三、逐题结果",
         "",
         "| ID | 问题 | 检索命中 | 回答命中 | 忠实度 | 相关性 | 命中来源 |",
         "|---|---|---|---|---|---|---|",
    ]
    # 逐题结果表格行：回答命中用✅/❌标记，来源只取前2个避免表格过宽
    for r in results:
        L.append(f"| {r['id']} | {r['question'][:28]}… | {r['c_ratio']:.0%} | "
                 f"{'✅' if r['a_ratio']>=0.5 else '❌'} | {r['faith']}/5 | {r['rel']}/5 | "
                 f"{','.join(r['rag']['sources'][:2])} |")

    # 追加逐题明细：问题、标准答案要点、RAG回答（截400字）、三维评估
    L += ["", "## 四、逐题明细", ""]
    for r in results:
        L += [f"### 问题 {r['id']}（{r['report']}）",
              f"**Q：** {r['question']}",
              f"**标准答案要点：** {r['answer_hint']}",
              f"**RAG回答：** {r['rag']['answer'][:400]}",
              "",
              f"评估：检索命中 {r['c_ratio']:.0%} | 回答命中 {r['a_ratio']:.0%} | "
              f"忠实度 {r['faith']}/5（{r['reason']}）",
              "", "---", ""]

    # 问题分析
    # 筛出回答未命中（<50%）的题，按"检索侧/生成侧"归因
    fails = [r for r in results if r["a_ratio"] < 0.5]
    L += ["## 五、检索结果存在的问题分析", ""]
    if not fails:
        L.append("本轮测试全部通过，未发现明显问题。")
    for r in fails:
        L.append(f"### 问题 {r['id']}（回答未命中）")
        L.append(f"- 现象：回答命中 {r['a_ratio']:.0%}，检索命中 {r['c_ratio']:.0%}")
        if r["c_ratio"] < 0.5:
            # 上下文本身就没命中 → 检索侧问题
            L.append("- 定性：**检索侧问题** —— 年报文本噪声大（表格转文字错位、重复段落），"
                     "关键数字所在分块未被 Top6 召回；改进方向：加强表格解析（工单03技术）、"
                     "提高 Top-K、引入重排器（工单06）。")
        else:
            # 上下文命中但答案没命中 → 生成侧问题
            L.append("- 定性：**生成侧问题** —— 上下文已含关键事实，但 7B 模型未正确引用"
                     "（多文档数字混淆或答非所问）；改进方向：更大参数 LLM、上下文压缩、自一致性投票。")
        L.append("")

    # 写出评估报告
    out = os.path.join(HERE, "测试及评估报告-工单07.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n总体: 检索命中 {hit_n}/{n} | 回答正确 {acc_n}/{n}")
    print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
