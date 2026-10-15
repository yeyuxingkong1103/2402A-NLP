"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。
由RAGFlow现有component_class自动发现，执行真实质检并输出路由。
"""
import asyncio,os
from pathlib import Path
from agent.component.base import ComponentBase,ComponentParamBase
from industrial_quality.assessment import assess
class DocumentQualityAssessmentParam(ComponentParamBase):
    def __init__(self):
        super().__init__();self.paths=[];self.config=None;self.output_dir='industrial_quality_reports'
    def check(self):
        if not isinstance(self.paths,list):raise ValueError('paths必须为列表')
class DocumentQualityAssessment(ComponentBase):
    component_name='DocumentQualityAssessment'
    async def _invoke_async(self,**kwargs):
        allowed=Path(os.getenv('DOCUMENT_QUALITY_ROOT','/ragflow/industrial_documents')).resolve()
        paths=[Path(p).resolve() for p in (self._param.paths or [str(allowed)])]
        if any(not p.is_relative_to(allowed) for p in paths):raise ValueError('超出质检允许目录')
        report=await asyncio.to_thread(assess,paths,self._param.config,self._param.output_dir)
        routes=[{'path':r['path'],'next_node':{'OCR':'OCRParser','PAGE_LEVEL':'PageTypeRouter','TEXT':'TextParser'}.get(r['route'],'ManualReview')} for r in report['files']]
        self.set_output('report',report);self.set_output('routes',routes)
