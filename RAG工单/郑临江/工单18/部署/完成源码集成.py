"""工单14-18：正式源码挂钩及带行号的差异记录，不用独立API冒充正式栈验收。"""
from pathlib import Path
import zipfile,difflib
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'工单17/研发/ragflow-main/api/apps/sdk/session.py';text=p.read_text(encoding='utf-8')
text=text.replace('async def chat_completion_openai_like(tenant_id, chat_id):','@__import__("common.bounded_chat",fromlist=["bounded_chat"]).bounded_chat\nasync def chat_completion_openai_like(tenant_id, chat_id):')
p.write_text(text,encoding='utf-8')
p=ROOT/'工单15/研发/ragflow-main/rag/nlp/search.py';text=p.read_text(encoding='utf-8')
needle='        filters = self.get_filters(req)'
text=text.replace(needle,needle+'''\n        # 工单15：请求明确的PDF物理页码成为过滤条件，避免仅命中“部件”等泛词。
        from common.visual_reference import visual_reference
        visual = visual_reference(req.get("question", ""))
        if visual["physical_page"] is not None:
            filters["page_num_int"] = [visual["physical_page"]]
''',1)
p.write_text(text,encoding='utf-8')
z=zipfile.ZipFile('D:/软件/QQ/data/RAG 工单1/14-17附件/ragflow-main.zip')
for num in range(14,19):
    repo=ROOT/f'工单{num}/研发/ragflow-main';diff=[]
    for p in repo.rglob('*.py'):
        relative=str(p.relative_to(repo)).replace('\\','/');source='ragflow-main/'+relative
        new=p.read_text(encoding='utf-8-sig');old=z.read(source).decode('utf-8-sig') if source in z.namelist() else ''
        if new!=old:diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
    (ROOT/f'工单{num}/优化/最终源码变更.patch').write_text(''.join(diff),encoding='utf-8')
    compile_errors=[]
    for p in repo.rglob('*.py'):
        relative=str(p.relative_to(repo)).replace('\\','/')
        if '工业' in p.name or any(token in relative for token in ['bounded_chat.py','visual_reference.py','industrial_page_ranges.py','document_quality_assessment.py','document_ingestion_router.py','quality_app.py','industrial_pdf_parser.py']):
            try:compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
            except Exception as exc:compile_errors.append(str(exc))
    print(num,'diff_lines',len(diff),'syntax_errors',compile_errors)
