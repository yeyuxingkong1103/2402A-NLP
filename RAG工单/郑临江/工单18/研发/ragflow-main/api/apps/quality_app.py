"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。正式RAGFlow Blueprint。"""
import asyncio,os
from pathlib import Path
from quart import request
from api.apps import login_required
from api.utils.api_utils import get_json_result
from industrial_quality.assessment import assess
@manager.route('/inspection',methods=['POST'])
@login_required
async def inspection():
    body=await request.get_json();allowed=Path(os.getenv('DOCUMENT_QUALITY_ROOT','/ragflow/industrial_documents')).resolve()
    paths=[Path(p).resolve() for p in body.get('paths',[str(allowed)])]
    if any(not p.is_relative_to(allowed) for p in paths):return get_json_result(code=403,message='超出质检允许目录')
    report=await asyncio.to_thread(assess,paths,None,os.getenv('DOCUMENT_QUALITY_OUTPUT','/ragflow/quality_reports'))
    return get_json_result(data=report)
