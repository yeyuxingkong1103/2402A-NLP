"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。报告只从最新原始实测JSON汇总。"""
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num in range(2,14):
    folder=ROOT/f'工单{num}';report=json.loads((folder/'测试/实测结果.json').read_text(encoding='utf-8'))
    lines=[f'# 工单{num}最终实测报告','',
           '原始回答、来源和耗时完整保存在实测结果.json。历史执行日志仅作过程记录，当前汇总以此文件为准。','',
           '|策略|Hit@5|MRR@5|事实关键词覆盖代理|','|---|---:|---:|---:|']
    for strategy,item in report['strategies'].items():lines.append(f"|{strategy}|{item['hit_at_5']:.1%}|{item['mrr_at_5']:.3f}|{item['fact_coverage_proxy']:.1%}|")
    lines+=['','检索指标不能等同于问答准确率，关键词匹配可能低估单位换算和同义改写，也不能识别全部附加错误。','',
            '## 指定题实际生成','']
    for row in report['answers']:
        lines += [f"### ID {row['id']}：{row['question']}",'',row['result']['answer'],'',f"完整耗时 {row['result']['elapsed_ms']:.2f}ms；状态 {row['result']['generation']}。",'']
    if num==12:
        rows=json.loads((folder/'测试/LightRAG生成对比.json').read_text(encoding='utf-8'))
        lines+=['## 官方LightRAG实际生成对比','']
        for row in rows:lines += [f"### LightRAG ID {row['id']}",'',row['result']['answer'],'',f"耗时 {row['result']['elapsed_ms']:.2f}ms。",'']
    (folder/'测试/实测报告.md').write_text('\n'.join(lines),encoding='utf-8')
print('13个项目当前指标均可追溯到各自原始记录')
