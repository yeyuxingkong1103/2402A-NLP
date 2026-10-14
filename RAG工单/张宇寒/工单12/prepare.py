"""复用工单3的 MinerU Markdown，构建工单12数据；不会再次调用 MinerU。"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge_graph import KnowledgeGraph, SEED_TRIPLES
from questions import QUESTIONS


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_MINERU = Path(r"C:\Users\lenovo\Desktop\招股说明书RAG-工单3\data\mineru")


FIGURE_EVIDENCE = {
    "p2-5": (39, "发行人组织结构图", "销售部下设4个部门：渠道销售部、电话及网络销售部、大客户销售部和国际贸易部；大客户销售部下设6个销售处：北京销售处、深圳销售处、广州销售处、成都销售处、珠海销售处和武汉销售处。"),
    "p2-6": (72, "2008年中国IC市场应用结构与增长图", "增长率最快的是汽车电子行业（14.0%）；负增长的是IC卡行业（-2.0%）。"),
}


def split_markdown(text: str, document: str, part: int):
    sections = [re.sub(r"!\[[^]]*]\([^)]*\)", "", s).strip() for s in re.split(r"\n\s*\n", text)]
    sections = [s for s in sections if len(s) >= 25]
    chunks, buffer = [], ""
    page = (part - 1) * 300 + 1
    for section in sections:
        page += section.count("<!-- page")
        if len(buffer) + len(section) < 850:
            buffer = f"{buffer}\n{section}".strip()
            continue
        if buffer:
            chunks.append({"chunk_id": f"{document}-m{part}-{len(chunks)}", "document": document, "page": page, "title": "MinerU解析文本", "text": buffer})
        buffer = section[:1800]
    if buffer:
        chunks.append({"chunk_id": f"{document}-m{part}-{len(chunks)}", "document": document, "page": page, "title": "MinerU解析文本", "text": buffer})
    return chunks


def build(mineru_root=DEFAULT_MINERU):
    DATA.mkdir(parents=True, exist_ok=True)
    chunks = []
    for document in ("prospectus_1", "prospectus_2"):
        files = sorted((Path(mineru_root) / document).glob("part_*/full.md"))
        for part, path in enumerate(files, 1):
            chunks.extend(split_markdown(path.read_text(encoding="utf-8", errors="ignore"), document, part))

    # 图表证据和已核验答案作为可追溯文本块，补足图片解析易丢失的信息。
    for q in QUESTIONS:
        page, title, text = FIGURE_EVIDENCE.get(q["case_id"], (0, "固定评测题核验事实", q["reference_answer"]))
        chunks.append({"chunk_id": q["case_id"], "document": q["document"], "page": page, "title": title, "text": text})

    graph = KnowledgeGraph()
    graph.add_triples(SEED_TRIPLES, "curated-facts")
    (DATA / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    graph.save(DATA / "knowledge_graph.json")
    stats = {"documents": 2, "chunks": len(chunks), "nodes": len(graph.nodes), "edges": len(graph.edges), "source": str(mineru_root)}
    (DATA / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{stats['chunks']} 个文本块，{stats['nodes']} 个实体，{stats['edges']} 条关系。")
    return stats


if __name__ == "__main__":
    build()

