"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。高置信度角色关系校正。"""
import json
import os
import shutil
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PYTHON='D:/an/envs/rags_/python.exe'
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
for num in range(2,14):
    folder=ROOT/f'工单{num}';config=json.loads((folder/'研发/project.json').read_text(encoding='utf-8'))
    if num!=13:shutil.copy2(ROOT/'工单13/研发/app/core.py',folder/'研发/app/core.py')
    p=folder/'研发/app/core.py';p.write_text('# 工单编号：'+config['ticket']+'\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
    command='from app.core import build_graph,INDEX;import json;build_graph(json.loads((INDEX/"chunks.json").read_text(encoding="utf-8")))'
    subprocess.run([PYTHON,'-c',command],cwd=folder/'研发',env=env,stdout=subprocess.DEVNULL,check=True)
    graph=json.loads((folder/'研发/data/index/graph.json').read_text(encoding='utf-8'))
    print('typed graph',num,len(graph['nodes']),len(graph['edges']),flush=True)
    (folder/'设计/图谱角色校正记录.md').write_text('# 图谱角色校正\n\n此前宽松董事长正则把“兼总经理”等职务词误标为人名。已限定显式冒号姓名；法定代表人限定发行人基本资料，避免关联方和中介混淆。重新抽取后保留真实原文与来源，不沿用错误关系。DISCLOSES_TOPIC仍只是文档主题提及，不是因果。\n',encoding='utf-8')
