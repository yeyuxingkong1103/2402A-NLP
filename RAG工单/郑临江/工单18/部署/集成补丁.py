"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。生成源码差异与入口。"""
import shutil,json,re,zipfile,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'工单18'
shutil.copytree(folder/'研发/document_quality_assessment',folder/'研发/ragflow-main/industrial_quality',dirs_exist_ok=True)
for num in [14,15]:
    repo=ROOT/f'工单{num}/研发/ragflow-main'
    if num==15:shutil.copy2(ROOT/'工单14/研发/ragflow-main/deepdoc/parser/industrial_pdf_parser.py',repo/'deepdoc/parser/industrial_pdf_parser.py')
    file=repo/'rag/app/naive.py';text=file.read_text(encoding='utf-8')
    needle='        name = layout_recognizer.strip().lower()'
    if 'industrial_page_preservation' not in text:
        text=text.replace(needle,"""        # 工单14/15：可选工业图文保真，保持默认解析器不变。
        if parser_config.get('industrial_page_preservation', False):
            from deepdoc.parser.industrial_pdf_parser import IndustrialPdfParser
            def industrial_parser(filename, binary=None, **options):
                return IndustrialPdfParser()(filename, binary=binary, **options)
            PARSERS['industrial'] = industrial_parser
            layout_recognizer = 'industrial'
"""+needle)
    file.write_text(text,encoding='utf-8')
repo=ROOT/'工单17/研发/ragflow-main';p=repo/'rag/svr/task_executor.py';text=p.read_text(encoding='utf-8')
# 该executor仅TOC使用，取消/提前返回路径无需创建；with确保任务完成或异常时释放。
text=text.replace('    executor = concurrent.futures.ThreadPoolExecutor()','    # 工单17：仅在TOC需要时创建，并由上下文确保线程池释放。')
text=text.replace('            toc_thread = executor.submit(build_TOC, task, chunks, progress_callback)',
    '            with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-toc") as toc_executor:\n                toc_thread = toc_executor.submit(build_TOC, task, chunks, progress_callback)')
p.write_text(text,encoding='utf-8')
z=zipfile.ZipFile('D:/软件/QQ/data/RAG 工单1/14-17附件/ragflow-main.zip')
for num in [14,15,17,18]:
    repo=ROOT/f'工单{num}/研发/ragflow-main';diff=[]
    files=['rag/app/naive.py','deepdoc/parser/industrial_pdf_parser.py'] if num in {14,15} else ['rag/svr/task_executor.py'] if num==17 else ['api/apps/quality_app.py','agent/component/document_quality_assessment.py']
    for relative in files:
        old=z.read('ragflow-main/'+relative).decode('utf-8') if 'ragflow-main/'+relative in z.namelist() else ''
        new=(repo/relative).read_text(encoding='utf-8')
        diff+=difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative)
    (ROOT/f'工单{num}/优化/ragflow.patch').write_text(''.join(diff),encoding='utf-8')
print('formal source patches generated')
