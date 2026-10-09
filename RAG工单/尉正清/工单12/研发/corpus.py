# 工单编号：人工智能NLP-RAG项目-LightRAG优化
"""两份招股说明书的解析 —— RAG 与 LightRAG **共用同一份文本**

为什么必须共用：本工单要比的是「图结构检索」与「扁平向量检索」。
如果两边各解析各的，检索结果的差异里就混进了「谁多读了一段」这个变量，
比出来的东西说不清是谁的功劳。所以解析只做一次，两种表征都用它的输出。

解析链路沿用 RAG 系统本身（`document.load_pdf` + `image_parser`），不另写一套：

    PDF ──document.load_pdf──┬─→ 文字页
                             └─→ 表格（带标题）
    PDF ──image_parser───────→ 图表页（多模态模型读图转文字）

**图表那一支对本工单是必需的**：16 个问题里第 5 题问组织结构图、
第 6 题问 IC 市场应用结构与增长图，答案只存在于图里，文本层没有。
"""
import time

from document import build_chunks, load_pdf
from image_parser import parse_pdf_images


def parse(doc, with_images=True, cache_dir=None, progress=None):
    """解析一份招股书。

    返回 (chunks, page_texts)：

    - `chunks`     RAG 向量库用：文字块 + 表格块 + 图表块，每块带页码与类型
    - `page_texts` LightRAG 用：整篇文本、逐页带页码标记（它自己会切块，
                   所以这里给全文而不是给块）
    """
    started = time.time()
    pages, tables = load_pdf(doc["path"])
    chunks = build_chunks(pages, tables)
    print(f"[解析] {doc['file']}：{len(pages)} 页正文 + {len(tables)} 张表 "
          f"→ {len(chunks)} 块（{time.time() - started:.1f}s）", flush=True)

    images = []
    if with_images:
        t0 = time.time()
        images = parse_pdf_images(doc["path"], cache_dir=cache_dir, progress=progress)
        if images:
            print(f"[解析] 图表页 {len(images)} 页（{time.time() - t0:.1f}s）", flush=True)
        chunks += images

    # 页码标记带上文档名：两份招股书页码各自从 1 开始，只写「第 22 页」
    # 在合一的语料里指代不明，检索回来也不知道是哪一份。
    page_texts = [f"【{doc['file']} 第 {p['page']} 页】\n{p['text']}" for p in pages]
    for img in images:
        page_texts.append(f"【{doc['file']}】{img['text']}")

    return chunks, page_texts


def full_text(page_texts):
    """把逐页文本拼成一篇，页与页之间留空行，便于 LightRAG 在页边界处切块。"""
    return "\n\n".join(page_texts)
