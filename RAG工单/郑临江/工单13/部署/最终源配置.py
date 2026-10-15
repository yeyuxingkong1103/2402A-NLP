"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。最终本地副本与中英文实体配置。"""
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num in range(2,14):
    folder=ROOT/f'工单{num}';config=json.loads((folder/'研发/project.json').read_text(encoding='utf-8'))
    if num!=13:shutil.copy2(ROOT/'工单13/研发/app/core.py',folder/'研发/app/core.py')
    core=folder/'研发/app/core.py';text=core.read_text(encoding='utf-8');header='# 工单编号：'+config['ticket']+'\n'
    if not text.startswith(header):core.write_text(header+text,encoding='utf-8')
    base=folder/'研发/app/base.py';text=base.read_text(encoding='utf-8')
    if 'net profit|净利润' not in text:
        text=text.replace('ALIASES = [','ALIASES = [\n    (r"net profit|净利润", "净利润 归属于 股东 利润"),')
    base.write_text(text,encoding='utf-8')
print('最终项目代码及英文财务术语支持已保存')
