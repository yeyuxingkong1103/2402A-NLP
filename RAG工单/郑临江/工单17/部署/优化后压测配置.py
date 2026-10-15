"""工单17：同样JMeter负载改用优化版本端口；原日志保持不变。"""
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
for kind in ['A-20','B-10']:
    p=next((BASE/'测试').glob('场景'+kind+'*.jmx'))
    text=p.read_text(encoding='utf-8').replace('>8817<','>8827<')
    p.with_name('优化后-'+p.name).write_text(text,encoding='utf-8')
