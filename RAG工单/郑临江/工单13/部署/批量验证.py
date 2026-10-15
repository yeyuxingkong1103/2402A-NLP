"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。逐项目真实验证，输出只写各项目目录。"""
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PYTHON='D:/an/envs/rags_/python.exe'
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
for num in [2,4,5,6,7,8,9,10,11,13]:
    folder=ROOT/f'工单{num}'
    if (folder/'测试/实测结果.json').exists():
        print('已有实测跳过',num,flush=True);continue
    for filename,log in [('回归测试.py','自动化测试日志.txt'),('评估.py','执行日志.txt')]:
        with (folder/'测试'/log).open('w',encoding='utf-8') as out:
            result=subprocess.run([PYTHON,str(folder/'测试'/filename)],cwd=folder,env=env,stdout=out,stderr=subprocess.STDOUT)
        print('完成',num,filename,'exit',result.returncode,flush=True)
