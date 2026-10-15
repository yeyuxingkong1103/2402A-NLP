"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。
批量初始化交付目录。仅用于创建快照，每个项目运行时只读本项目文件。
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('D:/软件/QQ/data/RAG 工单')
ATT = SOURCE / '附件'
TITLES = {
    1:'基于PDF文档的问答系统',2:'基于PDF文档的问答系统优化',3:'PDF表格解析及检索优化',
    4:'PDF图像内容解析及检索优化',5:'Query理解与多轮对话',6:'混合检索与重排',
    7:'金融年报功能测试及评估',8:'GraphRAG金融问答',9:'GraphRAG优化',
    10:'金融问答系统Docker部署',11:'Embedding模型微调',12:'LightRAG优化',13:'RAG性能瓶颈识别与优化'}
model=Path('C:/Users/lenovo/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def safe_name(info):
    try:
        return info.filename.encode('cp437').decode('gbk').split('/')[-1]
    except (UnicodeError,LookupError):
        return info.filename.split('/')[-1]


def main():
    import pymupdf
    task_files=sorted(SOURCE.glob('*.pdf'))
    for num in range(1,14):
        folder=ROOT/f'工单{num}'
        for stage in ['设计/参考资料','研发/app','研发/static','研发/data/documents','研发/data/index','测试','优化','部署']:
            (folder/stage).mkdir(parents=True,exist_ok=True)
        task=task_files[num-1]
        shutil.copy2(task,folder/'设计/参考资料'/task.name)
        text='\n\n'.join(f'PDF物理页 {i+1}\n{p.get_text()}' for i,p in enumerate(pymupdf.open(task)))
        (folder/'设计/工单原文.txt').write_text(text,encoding='utf-8')
        if num==1:
            continue
        for filename in ['__init__.py','main.py','server.py']:
            shutil.copy2(ROOT/'工单1/研发/app'/filename,folder/'研发/app'/filename)
        shutil.copy2(ROOT/'工单1/研发/app/core.py',folder/'研发/app/base.py')
        shutil.copy2(ROOT/'工单1/研发/static/index.html',folder/'研发/static/index.html')
        for name in ['requirements.txt','requirements-model.txt','.env.example']:
            shutil.copy2(ROOT/'工单1/研发'/name,folder/'研发'/name)
        config={'number':num,'title':TITLES[num],'ticket':'人工智能NLP-RAG-'+TITLES[num],
                'port':8764+num,'data_isolation':'所有数据、索引、反馈、配置、测试均在当前项目内',
                'embedding_model':'models/bge-small-zh-v1.5','default_strategy':'hybrid',
                'default_reranker':'tfidf','llm_model':'Qwen3.5:4B','llm_provider':'ollama',
                'llm_base_url':'http://127.0.0.1:11434','llm_timeout':120,
                'dataset':'annual_reports' if num in {7,8,9,10} else 'prospectus',
                'graph':num in {8,9,10,12},'multimodal':num in {4,5,6,11,12,13},'baseline_snapshot':'工单1实现的独立源码快照'}
        save(folder/'研发/project.json',config)
        # 权重复制到各工单内，不形成主目录的公共模型或公共知识库。
        shutil.copytree(model,folder/'研发/models/bge-small-zh-v1.5',dirs_exist_ok=True)
        if num in {7,8,9,10}:
            shutil.copy2(ATT/'sample_questions.pdf',folder/'设计/参考资料/sample_questions.pdf')
            with zipfile.ZipFile(ATT/'ccf_competition.zip') as archive:
                manifest=[]
                for info in archive.infolist():
                    if info.is_dir() or not info.filename.endswith('.pdf'):
                        continue
                    data=archive.read(info)
                    name=safe_name(info)
                    (folder/'研发/data/documents'/name).write_bytes(data)
                    manifest.append({'name':name,'sha256':hashlib.sha256(data).hexdigest(),'source':'ccf_competition.zip','bytes':len(data)})
                save(folder/'设计/附件来源清单.json',manifest)
        else:
            names=['招股说明书1.pdf'] if num==2 else ['招股说明书1.pdf','招股说明书2.pdf']
            for name in names:
                shutil.copy2(ATT/name,folder/'研发/data/documents'/name)
            save(folder/'设计/附件来源清单.json',[{'name':n,'sha256':hashlib.sha256((ATT/n).read_bytes()).hexdigest()} for n in names])
            shutil.copy2(ROOT/'工单1/测试/gold.json',folder/'测试/gold.json')
        # 各项目有自己的端口和启动脚本；不从其他工单导入代码或运行时数据。
        launch='''# 工单编号：人工智能NLP-RAG-NAME
param([string]$Python='D:\\an\\envs\\rags_\\python.exe')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..\\研发')
$env:PYTHONIOENCODING='utf-8'
$env:LLM_PROVIDER='ollama'
$env:LLM_BASE_URL='http://127.0.0.1:11434'
$env:LLM_MODEL='Qwen3.5:4B'
$env:LLM_TIMEOUT='120'
$env:EMBEDDING_MODEL_PATH=Join-Path (Get-Location) 'models\\bge-small-zh-v1.5'
if (-not (Test-Path -LiteralPath 'data\\index\\manifest.json')) { & $Python -m app.main build; if($LASTEXITCODE -ne 0){throw '建库失败'} }
Write-Host '浏览器访问：http://127.0.0.1:PORT'
& $Python -m app.main serve --port PORT
'''.replace('NAME',TITLES[num]).replace('PORT',str(8764+num))
        (folder/'部署/启动.ps1').write_text(launch,encoding='utf-8-sig')
        (folder/'一键启动.cmd').write_text('@echo off\r\nchcp 65001 >nul\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0部署\\启动.ps1"\r\npause\r\n',encoding='utf-8')
        print(f'工单{num}：独立资料与本地模型已准备',flush=True)
    # 复制已经验证过的招股文件1解析索引；工单2只使用这一独立快照。
    shutil.copytree(ROOT/'工单1/研发/data/index',ROOT/'工单2/研发/data/index',dirs_exist_ok=True)


if __name__=='__main__':
    main()
