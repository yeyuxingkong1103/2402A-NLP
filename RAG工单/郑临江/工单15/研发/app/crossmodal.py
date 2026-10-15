"""工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程。"""

import re
from .pipeline import Pipeline


def visual_intent(question):
    page = re.search(r"第\s*(\d+)\s*页", question)
    figure = re.search(r"图\s*(\d+)", question)
    return {
        "page": int(page.group(1)) if page else None,
        "figure": figure.group(1) if figure else None,
    }


def query(question, optimized=True):
    result = Pipeline().ask(question, "CN100347506C.pdf", optimized)
    result["intent"] = visual_intent(question)
    return result
