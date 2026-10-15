"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。离线复制，项目运行无跨目录依赖。"""
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num in [5,6,11,12,13]:
    shutil.copytree(ROOT/'工单4/研发/data/index',ROOT/f'工单{num}/研发/data/index',dirs_exist_ok=True)
for num in [8,9,10]:
    shutil.copytree(ROOT/'工单7/研发/data/index',ROOT/f'工单{num}/研发/data/index',dirs_exist_ok=True)
print('复制完成；每个工单数据均为独立物理文件')
