"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。"""
import json
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
from document_quality_assessment.assessment import assess,simhash
def test_core(tmp_path):
    (tmp_path/'a.txt').write_text('电话 13800138000 邮件 test@example.com',encoding='utf-8')
    (tmp_path/'b.txt').write_text('电话 13800138000 邮件 test@example.com',encoding='utf-8')
    report=assess([tmp_path],output_dir=tmp_path/'out')
    assert report['total_files']==2 and report['exact_duplicates'] and report['sensitive_review']
    assert (tmp_path/'out/quality_report.html').exists()
