# -*- coding: utf-8 -*-
"""
工单04：PDF图像内容解析及检索优化
工单编号：人工智能NLP-RAG-图像内容解析及检索优化
功能：
  1. 定位《招股说明书2.pdf》中含图表的页面（组织结构图 / IC市场增长图）；
  2. 整页渲染为图片，用多模态模型 Qwen2.5-VL 解析图像语义；
  3. 把图像描述文本向量化入库（zgs2_img_v1），实现"图内容可检索"；
  4. 对验收问题 id5（组织结构图）、id6（IC市场增长图）生成回答。
运行：python run_image.py
"""
import sys
import os
import json
import time

# 取当前脚本所在目录，用于拼接资源路径与输出路径
HERE = os.path.dirname(os.path.abspath(__file__))
# 公共模块目录（config/vector_store 等都在 00-公共模块 下），加入 sys.path 才能跨工单目录 import
COMMON = os.path.join(HERE, "..", "00-公共模块")
sys.path.insert(0, COMMON)

from config import PDF_ZGS2  # 招股说明书2.pdf 的绝对路径配置
from image_parser import find_pages_by_keyword, render_pages  # 关键词定位页码、整页渲染PNG
from ollama_client import client  # Ollama 本地模型客户端（chat / vl_describe）
from vector_store import VectorStore  # 自研 numpy 向量库（保存/加载/检索）

# 页面渲染 PNG 的输出目录
IMG_DIR = os.path.join(HERE, "page_images")
# 图像语义描述入库的索引名（与工单03的文本索引 zgs_all_v1 区分开）
IMG_INDEX = "zgs2_img_v1"

# 两个验收问题对应的图表定位关键词与解析提示词
TARGETS = [
    {
        "qid": 5,  # 验收问题 id5：组织结构图
        "keywords": ["发行人内部组织结构图"],  # 用该关键词在 PDF 全文中定位含图页码
        # VL 提示词：明确"只看组织结构图、不看股权结构图"，并要求逐项列出图中真实文字，防止模型泛泛描述
        "vl_prompt": "这一页包含'(二)发行人内部组织结构图'。请只针对内部组织结构图（不含上方的股权结构图），"
                     "列出：1、销售部由几个部门/分支构成；2、大客户销售部下设几个销售处，分别是哪些销售处；"
                     "3、组织结构图中其他主要部门名称。请逐项列出图中真实文字。",
    },
    {
        "qid": 6,  # 验收问题 id6：IC市场增长图
        # 给两个关键词做兜底，防止 PDF 内文写法有空格差异导致定位不到页
        "keywords": ["2008 年中国IC 市场应用结构与增长", "中国IC 市场应用结构与增长"],
        # VL 提示词：要求读出每个行业的数值与增长率，直接对应验收问题要的"增长最快/负增长"
        "vl_prompt": "这一页包含'2008年中国IC市场应用结构与增长(亿元)'图表。请读出图中每个行业/应用领域的名称、"
                     "市场规模数值和增长率数值，并指出增长率最快的行业和负增长的行业。",
    },
]

# 整页通用描述提示词（与上面针对问题的定向提示词互补，做图像知识库的补充）
DESC_PROMPT = "请详细描述这一页PDF中的图表内容（图表标题、图中各要素名称、数字、关系），用中文输出，供后续检索问答使用。"


def parse_and_describe():
    """图像解析：定位 → 渲染 → VL语义描述 → 入库"""
    # 幂等保护：索引已建过就直接跳过，避免重复跑几十秒的 VL 识别
    if VectorStore.load(IMG_INDEX) is not None:
        print(f"[图像索引] {IMG_INDEX} 已存在，跳过解析")
        return

    all_chunks = []  # 累积所有图像描述分块，最后一次性入库
    for t in TARGETS:
        # 按关键词在 PDF 中搜索含图页码
        pages = find_pages_by_keyword(PDF_ZGS2, t["keywords"])
        if not pages:  # 关键词一个都没匹配到则警告并跳过该目标
            print(f"[警告] 未找到关键词页: {t['keywords']}")
            continue
        pno = pages[0]  # 取第一个命中页（关键词通常只出现一次）
        print(f"[图像解析] 问题{t['qid']} → 第{pno}页，渲染+VL识别中（约30-60秒/页）...")
        # 把该页整页渲染为 PNG（多模态模型需要图片输入）
        imgs = render_pages(PDF_ZGS2, [pno], IMG_DIR)
        # 用定向 VL 提示词识别该页，得到与验收问题强相关的语义描述
        desc = client.vl_describe(imgs[0]["png_path"], t["vl_prompt"])
        print(f"  → 描述摘要: {desc[:120]}...")
        # 描述文本包一层头部标注后作为一个分块，page 记录来源页便于引用展示
        all_chunks.append({
            "text": f"【图像语义解析 | 第{pno}页 | 问题{t['qid']}相关图表】\n{desc}",
            "page": pno,
        })
        # 同时保存整页通用描述（作为图像知识库的补充）
        desc2 = client.vl_describe(imgs[0]["png_path"], DESC_PROMPT)
        all_chunks.append({
            "text": f"【图像语义解析 | 第{pno}页 | 整页图表描述】\n{desc2}",
            "page": pno,
        })

    # 保存原始描述文本（供人工检查）
    with open(os.path.join(HERE, "图像解析描述.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)  # ensure_ascii=False 保留中文可读

    store = VectorStore()
    store.add(all_chunks, source_name="招股说明书2.pdf(图像)")  # 来源名加"(图像)"便于检索结果里区分
    store.save(IMG_INDEX)  # 持久化图像语义索引


def main():
    parse_and_describe()  # 先确保图像索引建好（已存在则内部跳过）
    img_store = VectorStore.load(IMG_INDEX)  # 加载图像语义索引
    text_store = VectorStore.load("zgs_all_v1")  # 加载工单03建的招股书全文文本索引

    # 两个验收问题（与 TARGETS 的 qid 一一对应）
    questions = [
        {"id": 5, "question": "武汉力源信息技术股份有限公司组织结构图中，销售部有几个部门构成，其中大客户销售部有几个销售处构成？"},
        {"id": 6, "question": "武汉力源信息技术股份有限公司招股意向书中，从2008年中国IC市场应用结构与增长图中可以看出，增长率最快的是哪个行业？负增长的是哪个行业？"},
    ]

    results = []  # 收集每题的问答结果，最后写报告
    for q in questions:
        print(f"\n[检索问答] {q['question']}")
        # 图像库优先 + 文本库补充：图像2条 + 文本5条，合并取前5
        img_hits = img_store.search(q["question"], top_k=2)
        text_hits = text_store.search(q["question"], top_k=5)
        # 按相似度分数降序合并两路结果，截取前5作为最终上下文候选
        hits = sorted(img_hits + text_hits, key=lambda h: -h["score"])[:5]

        # 相邻分块扩展：命中块若以残句开头（如"、武汉、珠海各设有..."），
        # 把同页前一个分块拼进上下文，避免句子被分块边界切断
        ctx_texts = []
        for h in hits:
            t = h["text"]
            # 仅对文本库命中块做扩展（图像描述块没有"上一块"的概念）
            if h.get("source") and h["text"] in text_store.texts:
                idx = text_store.texts.index(h["text"])  # 找到命中块在库中的下标
                # 前一块存在且与命中块同页，才拼接（跨页拼接会引入无关内容）
                if idx > 0 and text_store.metadatas[idx - 1].get("page") == h.get("page"):
                    prev = text_store.texts[idx - 1]
                    t = prev[-150:] + t  # 拼接上一块尾部，补全被切断的句子
            ctx_texts.append(t)
        # 拼装最终上下文：每个片段标注序号、来源库和页码，方便模型引用
        context = "\n\n".join(
            f"[片段{i+1} | {hits[i].get('source','图像库')} 第{hits[i].get('page','?')}页]\n{t}"
            for i, t in enumerate(ctx_texts))
        from rag_engine import QA_PROMPT  # 复用工单01的统一问答提示词模板
        # 低 temperature(0.1) 保证事实型回答稳定，限制500 token 防止跑题长篇
        answer = client.chat([{"role": "user",
                               "content": QA_PROMPT.format(context=context, question=q["question"])}],
                             temperature=0.1, num_predict=500)
        # 记录问题、答案及前5个检索来源（分数保留3位小数）
        results.append({"id": q["id"], "question": q["question"], "answer": answer,
                        "sources": [{"page": h["page"], "source": h["source"],
                                     "score": round(h["score"], 3)} for h in hits]})
        print(f"  A: {answer[:200]}")

    # 写 Markdown 验收报告：逐题列出问答与检索来源
    with open(os.path.join(HERE, "检索结果-工单04.md"), "w", encoding="utf-8") as f:
        f.write("# 工单04：图像内容解析及检索结果\n\n"
                f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}  |  多模态模型：qwen2.5vl:7b\n\n")
        for r in results:
            f.write(f"### 问题 {r['id']}\n**Q：** {r['question']}\n\n**A：** {r['answer']}\n\n"
                    "检索来源：" + " | ".join(f"{s['source']}第{s['page']}页({s['score']})" for s in r["sources"][:3]) + "\n\n---\n\n")
    print("\n已生成: 检索结果-工单04.md")


if __name__ == "__main__":
    main()
