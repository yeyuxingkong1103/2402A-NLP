"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。最后的独立专项回归及环境核验。"""
import json
import os
import shutil
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PYTHON='D:/an/envs/rags_/python.exe'
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
for num in range(2,14):
    folder=ROOT/f'工单{num}'
    for name in ['core.py','base.py','server.py','main.py']:
        path=folder/'研发/app'/name;text=path.read_text(encoding='utf-8')
        config=json.loads((folder/'研发/project.json').read_text(encoding='utf-8'))
        header='# 工单编号：'+config['ticket']+'\n'
        if not text.startswith(header):path.write_text(header+text,encoding='utf-8')
    # 改进的图谱规则在各独立知识库重新运行，不复制别人的图谱运行时数据。
    build='from app.core import build_graph,INDEX;import json;build_graph(json.loads((INDEX/"chunks.json").read_text(encoding="utf-8")))'
    subprocess.run([PYTHON,'-c',build],cwd=folder/'研发',env=env,stdout=subprocess.DEVNULL,check=True)
    with (folder/'测试/自动化测试日志.txt').open('w',encoding='utf-8') as out:
        result=subprocess.run([PYTHON,str(folder/'测试/回归测试.py')],cwd=folder,env=env,stdout=out,stderr=subprocess.STDOUT)
    print('final regression',num,result.returncode,flush=True)
    if num==2:
        shutil.copy2(ROOT/'工单1/测试/实测评估结果.json',folder/'测试/工单1基线独立快照.json')
    if num in {8,9}:
        baseline=ROOT/f'工单{7 if num==8 else 8}/测试/实测结果.json'
        shutil.copy2(baseline,folder/'测试/前置工单对照独立快照.json')
print('Final local regressions done',flush=True)
