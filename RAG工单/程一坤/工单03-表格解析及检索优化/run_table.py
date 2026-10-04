# -*- coding: utf-8 -*-
"""
工单03：表格解析检索演示
工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
功能：对工单要求的14个问题（力源4题 + 兴图新科10题）在双文档知识库上检索，
     输出答案 + 检索分数 + 来源页码 →《检索结果-工单03.md》
运行：python run_table.py
"""
import sys
import os
import json
import time

# 当前脚本所在目录
HERE = os.path.dirname(os.path.abspath(__file__))
# 公共模块目录路径并加入模块搜索路径
COMMON = os.path.join(HERE, "..", "00-公共模块")
sys.path.insert(0, COMMON)

# 自研向量库：加载 zgs_all_v1 双文档索引
from vector_store import VectorStore
# 公共 QA 提示词模板（上下文+问题 → 指令）
from rag_engine import QA_PROMPT
# B轨混合检索组件：BM25 索引 / 向量+BM25加权混合检索 / 重排
from hybrid_retriever import BM25Index, hybrid_search, apply_rerank
# Ollama 客户端：直接调用 qwen2.5:7b-instruct 生成答案
from ollama_client import client

TOP_K = 12  # 最终送入 LLM 的候选片段数上限


def ask(store, bm25, question):
    """双轨检索最终版：
    A轨=查询扩展多路向量召回（解决"国防客户销售额≈军用领域收入"类语义等价缺口）
    B轨=混合检索（向量+BM25加权，解决表格块向量相似度低的问题）
    各取6席去重合并。修复记录：此前两轨分数量纲不一致（向量原始分0.6~0.78 vs
    混合归一分≤0.65），导致BM25第1名真值块被挤出候选，此处按轨道配额各取6席。"""
    # 函数内延迟导入查询优化组件（expand_query/multi_retrieve 与工单02共用）
    from query_optimize import expand_query, multi_retrieve
    t0 = time.time()
    # LLM 把原问题改写成 3 个等价问法，弥补问法与文档用词的语义缺口
    variants = expand_query(question, n=3)
    # A轨：每个问法各自召回，RRF 融合取6席（fetch_k=10 是每路预取候选数，供融合筛选）
    a_hits = multi_retrieve(store, variants, top_k=6, fetch_k=10)
    # B轨：向量+BM25 混合检索取6席；vec_w/ft_w 为两路权重（BM25占大头，补表格数字的字面匹配）
    # fetch_k=30 预取更多候选供加权排序筛选
    b_hits = hybrid_search(store, bm25, question, mode="hybrid",
                           vec_w=0.35, ft_w=0.65, top_k=6, fetch_k=30)
    # 两轨合并去重：以分块文本为唯一键（seen 集合判重），A轨结果在前优先保留
    seen, hits = set(), []
    for h in a_hits + b_hits:
        if h["text"] not in seen:
            seen.add(h["text"])
            hits.append(h)
    # 截断到 TOP_K=12，防止上下文过长导致 prompt 超限/生成质量下降
    hits = hits[:TOP_K]
    # 拼装上下文：每片段带来源文档、页码、相似度分数，便于生成时参考与结果溯源
    context = "\n\n".join(
        f"[片段{i+1} | {h['source']} 第{h['page']}页 | 相似度{h['score']:.3f}]\n{h['text']}"
        for i, h in enumerate(hits))
    # 调用 LLM 生成；prompt 末尾追加两条防错指令：①优先逐字匹配的关键句 ②警惕口径混淆
    # （踩坑：LLM 曾把"前五名客户销售额/合计"误答成"军用领域收入"，temperature=0.0 保证稳定）
    answer = client.chat([{"role": "user", "content": QA_PROMPT.format(context=context, question=question) +
                            "\n注意：请先逐条检查上下文片段，找出与问题最直接对应的句子（几乎逐字匹配问题关键词的表述）并以其为准作答；"
                            "警惕口径混淆：'来自军用领域的收入'≠'总计/合计'≠'前五名客户销售额'。"}],
                         temperature=0.0, num_predict=600)
    # 返回答案、来源（页码/文档/分数保留3位小数）、总耗时
    return {"answer": answer,
            "sources": [{"page": h["page"], "source": h["source"], "score": round(h["score"], 3)} for h in hits],
            "time_cost": round(time.time() - t0, 2)}


def main():
    # 加载工单03构建的双文档统一索引；不存在则提示先运行 build_index_all.py
    store = VectorStore.load("zgs_all_v1")
    if store is None:
        print("[错误] 请先运行 build_index_all.py")
        return
    # 尝试加载已持久化的 BM25 索引（B轨检索依赖）
    bm25 = BM25Index.load("zgs_all_v1")
    if bm25 is None:
        # 首次运行：基于当前索引的全部分块与元数据构建 BM25 倒排索引并持久化
        print("[BM25] 首次构建 ...")
        bm25 = BM25Index().build(store.texts, store.metadatas)
        bm25.save("zgs_all_v1")

    # 14 个验收问题（力源4题来自工单03要求 + 兴图新科10题沿用）
    # 公共评测集中的兴图新科 10 题
    with open(os.path.join(COMMON, "ground_truth.json"), encoding="utf-8") as f:
        gt = json.load(f)["questions"]
    # 工单03新增的武汉力源 4 题（针对招股说明书2）
    liyuan_qs = [
        {"id": 1, "question": "武汉力源信息技术股份有限公司本次发行股数是多少，占发行后总股本的比例是多少？"},
        {"id": 2, "question": "武汉力源信息技术股份有限公司本次募集资金拟投资哪些项目？"},
        {"id": 3, "question": "与武汉力源信息技术股份有限公司存在控制关系的关联方是谁，持股比例和本公司关系是什么？"},
        {"id": 4, "question": "与武汉力源信息技术股份有限公司不存在控制关系的关联方企业有哪些？"},
    ]
    # 合并为完整 14 题评测集（力源在前，兴图新科在后）
    questions = liyuan_qs + gt

    # 报告头部：标题 + 生成时间 + 知识库说明
    L = ["# 工单03：表格解析及检索优化 —— 检索结果",
         "",
         f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  |  知识库：招股说明书1+2（含表格解析）",
         ""]
    # 逐题执行双轨检索问答并写入报告
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"[{i}/{len(questions)}] {q}")
        try:
            r = ask(store, bm25, q)
        except Exception as e:
            # 单题异常不中断整批实验，错误信息直接记入报告
            r = {"answer": f"ERROR: {e}", "sources": [], "time_cost": "-"}
        L += [
            f"### 问题 {item['id']}",
            f"**Q：** {q}",
            f"**A：** {r['answer']}",
            "",
            # 检索精确度行：列出前3个来源（文档+页码+分数），展示双轨检索的可溯源证据
            "检索精确度：" + " | ".join(
                f"{s['source']}第{s['page']}页({s['score']})" for s in r["sources"][:3]),
            f"耗时：{r['time_cost']}s",
            "", "---", "",
        ]
    # 报告输出到工单03目录
    out = os.path.join(HERE, "检索结果-工单03.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"已生成: {out}")


if __name__ == "__main__":
    main()
