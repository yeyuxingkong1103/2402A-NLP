"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。评估保留版本，不重复改写历史输出。"""
import json
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PYTHON='D:/an/envs/rags_/python.exe'
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
for num in [8,9]:
    folder=ROOT/f'工单{num}'
    with (folder/'测试/最终独立复测日志.txt').open('w',encoding='utf-8') as out:
        result=subprocess.run([PYTHON,str(folder/'测试/评估.py')],cwd=folder,env=env,stdout=out,stderr=subprocess.STDOUT)
    print('Final graph generation',num,result.returncode,flush=True)
