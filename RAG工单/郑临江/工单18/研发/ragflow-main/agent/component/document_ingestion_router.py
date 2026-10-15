"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。实际下游分流解析节点。"""
import asyncio,os
from pathlib import Path
import pymupdf
from agent.component.base import ComponentBase,ComponentParamBase
class DocumentIngestionRouterParam(ComponentParamBase):
    def __init__(self):super().__init__();self.quality_node='document_quality:0';self.max_documents=1
    def check(self):
        if self.max_documents<1:raise ValueError('max_documents必须为正数')
class DocumentIngestionRouter(ComponentBase):
    component_name='DocumentIngestionRouter'
    async def _invoke_async(self,**kwargs):
        routes=self._canvas.get_variable_value(self._param.quality_node+'@routes')
        results=[]
        for route in routes[:self._param.max_documents]:
            path=Path(route['path']);allowed=Path(os.getenv('DOCUMENT_QUALITY_ROOT','/ragflow/industrial_documents')).resolve()
            if not path.resolve().is_relative_to(allowed):raise ValueError('路径超出入库目录')
            if route['next_node']=='ManualReview':results.append({**route,'status':'manual_review'});continue
            if route['next_node'] in {'OCRParser','PageTypeRouter'}:
                from deepdoc.parser import PdfParser
                sections,tables=await asyncio.to_thread(PdfParser(),str(path))
                results.append({**route,'status':'parsed','sections':len(sections),'figures_tables':len(tables)})
            else:
                def read():
                    with pymupdf.open(path) as doc:return sum(len(p.get_text()) for p in doc)
                results.append({**route,'status':'parsed','characters':await asyncio.to_thread(read)})
        self.set_output('parse_results',results)
