# 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
# 【PDF解析组件 · pdf_parser.py】将招股说明书解析为带页码、可溯源的文本/表格片段
import json
import pickle
from pathlib import Path
from typing import Iterator, List, Dict
import fitz  # PyMuPDF
import pdfplumber

import config


def _extract_page_text(pdf_path: Path) -> Iterator[Dict]:
    """逐页提取纯文本，返回 {page, text}"""
    doc = fitz.open(pdf_path)
    for page_no in range(doc.page_count):
        page = doc[page_no]
        text = page.get_text("text") or ""
        yield {"page": page_no + 1, "text": text.strip()}
    doc.close()


def _extract_tables(pdf_path: Path) -> Dict[int, List]:
    """
    提取每页表格，返回 {page: [table, ...]}
    表格提取结果按 PDF 做 pickle 缓存，避免重复跑 548 页
    """
    cache_path = config.tables_cache_for(pdf_path)
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    tables_by_page: Dict[int, List] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables() or []
            except Exception as e:
                print(f"[WARN] page {idx} 表格解析失败: {e}")
                tables = []
            if tables:
                tables_by_page[idx] = tables

    cache_path.write_bytes(pickle.dumps(tables_by_page))
    return tables_by_page


def _table_to_text(table: List[List[List[str]]]) -> str:
    """将表格转为可被 Embedding 的扁平文本"""
    lines = []
    for row in table:
        cells = [str(c).strip() if c else "" for c in row]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_pdf(pdf_path: Path, cache: bool = True) -> List[Dict]:
    """
    解析 PDF，返回 [{page, text, has_table}, ...]
    cache=True 时优先读取 .jsonl 缓存，避免重复解析 548 页大文件
    """
    pdf_path = Path(pdf_path)
    cache_path = config.jsonl_cache_for(pdf_path)

    # 缓存命中
    if cache and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    pages = list(_extract_page_text(pdf_path))
    tables = _extract_tables(pdf_path) if config.PDF_TABLEExtract else {}

    result: List[Dict] = []
    for item in pages:
        page = item["page"]
        text = item["text"]
        # 把表格内容拼到正文后，便于一整块检索
        if page in tables:
            for t in tables[page]:
                text += "\n[表格]\n" + _table_to_text(t)
        if text.strip():
            result.append({"page": page, "text": text, "has_table": page in tables})

    # 写缓存
    with cache_path.open("w", encoding="utf-8") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] 解析完成: {pdf_path.name} 共 {len(result)} 页有文字")
    return result


if __name__ == "__main__":
    # 自测：解析招股说明书并打印前两页摘要
    data = parse_pdf(config.SOURCE_PDF)
    print(f"解析得到 {len(data)} 个页面记录")
    for d in data[:2]:
        print(f"--- page {d['page']} (has_table={d['has_table']}) ---")
        print(d["text"][:200])
        print()

# ====================================================================
# 技术备注：
# 1. RAG：本模块属于 RAG 的"离线入库"环节，输出供后续切分、向量化使用。
# 2. PDF 解析选择 PyMuPDF+PDFPlumber 双引擎：
#    - PyMuPDF 速度快，文字层提取稳定
#    - PDFPlumber 表格识别准，弥补前者表格能力弱
# 3. Transformer：本模块不涉及，但产出的文本块将作为 Embedding Transformer 的输入。
# 4. Fine-tuning：如需进一步提升表格检索效果，可使用 LayoutLM/TableLM 微调。
# ====================================================================
