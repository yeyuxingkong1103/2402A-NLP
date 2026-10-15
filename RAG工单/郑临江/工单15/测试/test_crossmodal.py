"""工单编号：人工智能NLP-RAG-优化技术图纸与文本的跨模态检索流程"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'研发'))
from app.crossmodal import visual_intent
def test_figure_intent():
    assert visual_intent('第11页图3中编号13位置')['page']==11
    assert visual_intent('第11页图3中编号13位置')['figure']=='3'
