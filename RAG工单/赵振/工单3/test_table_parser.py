"""工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化。"""

import json
import os
import sys
import time
from pathlib import Path

from rag import table_chunks


ROOT = Path(__file__).parent
PDF = Path(os.getenv("TASK03_PDF2", r"C:\Users\ZhuanZ\Downloads\RAG 工单\RAG 工单\附件\招股说明书2.pdf"))
QUESTIONS = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))[:4]


def main():
    started = time.perf_counter()
    chunks = table_chunks(PDF)
    issuance = [c for c in chunks if c["table_group"] == "issuance" and c["page"] == 22]
    projects = [c for c in chunks if c["table_group"] == "fund_projects" and c["page"] == 22]
    control = [c for c in chunks if c["table_group"] == "control_relations" and c["page"] == 157]
    entities = [c for c in chunks if c["table_group"] == "related_entities" and c["page"] == 157]

    assert any("1,670 万股" in c["text"] and "25.04%" in c["text"] for c in issuance)
    assert len(projects) == 5
    assert len(control) == 1 and "42.35%" in control[0]["text"]
    assert len(entities) == 7
    assert not any(c["table_group"] == "related_entities" and c["page"] == 158 for c in chunks)

    result = {
        "pdf": PDF.name,
        "pages": 350,
        "question_count": len(QUESTIONS),
        "table_row_count": len(chunks),
        "checks": {
            "issuance_shares_and_ratio": {"pass": True, "page": 22, "rows": len(issuance)},
            "fund_projects": {"pass": True, "page": 22, "rows": len(projects)},
            "controlling_party": {"pass": True, "page": 157, "rows": len(control)},
            "noncontrolling_parties": {"pass": True, "page": 157, "rows": len(entities)},
            "historical_entities_excluded": {"pass": True, "page": 158},
        },
        "seconds": round(time.perf_counter() - started, 3),
    }
    (ROOT / "table_parser_test_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
