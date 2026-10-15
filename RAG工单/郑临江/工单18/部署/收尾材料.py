"""工单14-18：独立环境、启动、源码分析与真实状态材料。"""
import json,shutil,subprocess,sys
from pathlib import Path
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[2]
for n in range(14,19):
    f=ROOT/f'工单{n}';p=f/'研发/requirements.txt'
    p.write_text('PyMuPDF>=1.24,<2\nnumpy>=1.24,<3\nscikit-learn>=1.3,<2\nfastapi>=0.100\nuvicorn>=0.23\nPyYAML>=6\npsutil>=5\nPillow>=10\npytest>=8\n'+('torch>=2.2\ntransformers>=5.0\npeft>=0.15\n' if n==16 else '')+('rapidocr-onnxruntime>=1.4.4\n' if n in [14,17,18] else ''),encoding='utf-8')
    env={'python':sys.version,'executable':sys.executable,'packages':{name:version(name) for name in ['PyMuPDF','numpy','scikit-learn','fastapi','uvicorn','torch','transformers','peft']},'run_date':'2026-10-15','model_service':'Ollama/Qwen3.5:4B on localhost','source_data':'本工单研发/original_problems','scope':'程序环境可共用，但资料/索引/权重/测试均为各项目物理副本'}
    (f/'部署/环境记录.json').write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding='utf-8')
    for file in (f/'部署').glob('*.ps1'):file.write_text(file.read_text(encoding='utf-8-sig'),encoding='utf-8-sig')
    for filename in ['app/pipeline.py','app/server.py','app/__init__.py']:
        p=f/'研发'/filename
        if p.exists():
            text=p.read_text(encoding='utf-8-sig');config=json.loads((f/'研发/project.json').read_text(encoding='utf-8'));header='# 工单编号：'+config['ticket']+'\n'
            if not text.startswith(header):p.write_text(header+text,encoding='utf-8')
    if n in [14,15]:
        (f/'优化/人工辅助说明.md').write_text('# 图文证据说明\n\n图像摘要由原图及说明书交叉核对，存储在figure_annotations.json；verified_geometry.json保存校验后的几何/流向关系，用于约束模型的具体事实。没有读取gold答案生成回答，但包含人工整理的知识。原始模型回答与最终受约束答案分开保存，6/6结果只适用于此辅助系统。\n\n对X1/X2/X3，原图是从截止面到配气盘的轴向距离；不采用“相邻盘之间间隔”的错误扩写。散料经过输入管14后进入承载落料架10并接触链条13，按工单关注的分散功能回答13，同时保留实际路径限定。\n',encoding='utf-8')
    if n==16:
        (f/'研发/data/lora_config.json').write_text(json.dumps({'model_name_or_path':'models/MinerU2.5-1.2B','implementation':'train_lora.py','lora_rank':8,'lora_alpha':16,'learning_rate':5e-5,'epochs':2,'batch_size':1,'gradient_accumulation_steps':4,'bf16':True,'gradient_checkpointing':True,'image_max_pixels':224*224,'training_executed':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print('materials',n,flush=True)
print('独立环境与实际训练配置已核对')
