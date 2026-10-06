# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【文本切分组件 · text_splitter.py】将整页文本切分为 512 字符语义块，保留页码元数据
from typing import List, Dict
try:
    # 新版 langchain 已将 splitter 拆分到独立包
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # 老版本回退

import config


def _make_splitter() -> RecursiveCharacterTextSplitter:
    """构建递归字符切分器"""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        keep_separator=True,
    )


def split_pages(pages: List[Dict]) -> List[Dict]:
    """
    输入: pdf_parser.parse_pdf 输出 [{page, text, has_table}, ...]
    输出: [{chunk_id, page, text}, ...]
    每个文本块大小约 512 字符，重叠 64 字符，便于跨块语义不丢
    """
    splitter = _make_splitter()
    chunks: List[Dict] = []
    cid = 0
    for page in pages:
        parts = splitter.split_text(page["text"])
        for p in parts:
            p = p.strip()
            if not p:
                continue
            chunks.append({
                "chunk_id": cid,
                "page": page["page"],
                "text": p,
            })
            cid += 1
    print(f"[OK] 切分完成: 共 {len(chunks)} 个文本块")
    return chunks


if __name__ == "__main__":
    import pdf_parser
    pages = pdf_parser.parse_pdf(config.SOURCE_PDF)
    chunks = split_pages(pages)
    for c in chunks[:3]:
        print(f"--- chunk {c['chunk_id']} page {c['page']} ---")
        print(c["text"][:200])
        print()

# ====================================================================
# 技术备注：
# 1. RAG：切分质量直接决定检索精度。过短丢失语义，过长稀释关键信息。
# 2. RecursiveCharacterTextSplitter：按 separators 顺序递归切分，优先在段落/句号处断开。
# 3. 64 字符 overlap 保证跨块边界信息不丢失。
# 4. Fine-tuning：可引入语义切分（如基于句法树）进一步优化。
# ====================================================================
