"""工单编号：人工智能NLP-RAG-修复低质量工业PDF的解析与信息丢失"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
from app.pipeline import Pipeline
def test_evidence_and_page_constraint():
    p=Pipeline();p.reload();hits=p.retrieve('在文件中第7页的图片中，部件4相对于部件5的位置关系？','CN100342976C.pdf',True)
    assert hits and all(x['pdf_page']==7 for x in hits)
