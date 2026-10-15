"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。各项目独立解压附件。"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parents[2]
SRC=Path('D:/软件/QQ/data/RAG 工单1')
titles={14:'修复低质量工业PDF的解析与信息丢失',15:'优化技术图纸与文本的跨模态检索流程',16:'调试与微调专用视觉语言模型',17:'解决API服务并发瓶颈与资源泄漏',18:'实现文档质量评估Skill并集成至智能体工作流'}
for n in range(14,19):
    folder=ROOT/f'工单{n}'
    for sub in ['设计/参考资料','研发/app','研发/data','研发/models','研发/static','测试','优化','部署']:(folder/sub).mkdir(parents=True,exist_ok=True)
    task=next(SRC.glob(f'*项目-{n}-*.pdf'));shutil.copy2(task,folder/'设计/参考资料'/task.name)
    (folder/'设计/工单原文.txt').write_text('\n'.join(f'PDF页{i+1}\n'+p.get_text() for i,p in enumerate(pymupdf.open(task))),encoding='utf-8')
    for filename in ['original_problems.zip','ragflow-main.zip']:
        with zipfile.ZipFile(SRC/'14-17附件'/filename) as z:
            for member in z.infolist():
                if member.is_dir():continue
                target=(folder/'研发'/member.filename).resolve()
                if not target.is_relative_to((folder/'研发').resolve()):raise ValueError(member.filename)
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as source,target.open('wb') as dest:shutil.copyfileobj(source,dest)
    # 原始问题与答案保存在测试/训练源，在线检索代码不导入答案字段。
    question_file=folder/'研发/original_problems/questions.jsonl'
    rows=[json.loads(line) for line in question_file.read_text(encoding='utf-8').splitlines()]
    docs=list((folder/'研发/original_problems/documents').glob('*.pdf'))
    config={'number':n,'title':titles[n],'ticket':'人工智能NLP-RAG-'+titles[n]+'工单','port':8800+n,
      'documents':'original_problems/documents','questions':'original_problems/questions.jsonl','llm_url':'http://127.0.0.1:11434',
      'llm_model':'Qwen3.5:4B','target_document':'CN100342976C.pdf' if n==14 else 'CN100347506C.pdf',
      'source_dataset_documents':len(docs),'source_questions':len(rows),'data_isolation':'本项目目录内物理副本；运行不引用其他工单路径'}
    (folder/'研发/project.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    gold=[{**r,'reference':next((s[3:].strip() for s in r['options'] if s.startswith(r['answer']+'.')),r['answer'])} for r in rows if r['document']==config['target_document']]
    (folder/'测试/gold.json').write_text(json.dumps(gold,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'attachments':[{ 'name':x,'sha256':hashlib.sha256((SRC/'14-17附件'/x).read_bytes()).hexdigest()} for x in ['original_problems.zip','ragflow-main.zip']],
       'documents':len(docs),'questions':len(rows),'source_version':'ragflow0.23.1 bundled snapshot'}
    (folder/'设计/数据来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('工单',n,'独立解压',len(docs),len(rows),flush=True)
