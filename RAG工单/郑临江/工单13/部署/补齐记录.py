"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。补齐专项说明与最终复现记录。"""
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num in [9,12]:
    folder=ROOT/f'工单{num}';p=folder/'测试/RAGAS实测结果.json'
    if not p.exists():continue
    records=json.loads(p.read_text(encoding='utf-8'));summary={}
    for strategy in {x['strategy'] for x in records}:
        summary[strategy]={}
        for metric in ['context_precision','context_recall']:
            values=[x['metrics'][metric] for x in records if x['strategy']==strategy and isinstance(x['metrics'].get(metric),(int,float))]
            summary[strategy][metric]=statistics.mean(values) if values else None
            summary[strategy][metric+'_valid_samples']=len(values)
    (folder/'测试/RAGAS指标汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text='# RAGAS实际指标对照\n\n评审模型Qwen3.5:4B，ragas0.4.3；依据原始逐题指标取均值，错误项不填合格分数。\n\n|策略|ContextPrecision|ContextRecall|有效样本|\n|---|---:|---:|---:|\n'
    for strategy,row in summary.items():text+=f"|{strategy}|{row['context_precision']}|{row['context_recall']}|{row['context_precision_valid_samples']}/{row['context_recall_valid_samples']}|\n"
    if num==9:
        row=summary.get('graph_optimized',{})
        passed=row.get('context_precision',0)>=.8 and row.get('context_recall',0)>=.9
        text+='\n平均精度0.8、召回0.9阈值：'+('通过' if passed else '未通过')+'。单题可能未达标，尤其跨公司复杂问句需继续优化；平均达标不能掩盖逐题错误。\n'
    text+='\n标准事实关键词覆盖不是这些指标。评审模型存在不确定性，人工需复核数字、单位及引用归因。\n'
    (folder/'测试/RAGAS对比报告.md').write_text(text,encoding='utf-8')
for num in range(2,14):
    folder=ROOT/f'工单{num}'
    text='# 过程与问题记录\n\n'
    text+='从本工单PDF提取需求，离线复制前置实现，所有运行时数据放本工单内。源文件哈希保留，不从主目录读公用知识库。\n\n'
    if num in {7,8,9,10}:text+='年报find_tables对复杂图形页耗时异常，改成全文全页+摘要表格的有界解析。原eval_question.md未提供，按sample_questions和年报补建具名题；明确标注不是原作者问题文件。\n\n'
    if num in {8,9,10,12}:text+='宽松人名正则曾误把“兼总经理”等标成人名；最终限定明确职务冒号姓名/发行人基本资料。共现主题关系不推断因果或持股。\n\n'
    if num==10:text+='bridge容器不能访问只绑定Windows本机的Ollama，改WSL host网络；实际重启和数据卷持久化检查通过。\n\n'
    if num==11:text+='SentenceTransformers6.1预处理包含非张量字段，训练只迁移Tensor到GPU后运行成功。按验证集选择checkpoint，测试集8题无训练页交叉。\n\n'
    if num==12:text+='官方Tokenizer新增TokenSpan契约，使用官方包装器适配本地tokenizer。初始总上下文预算过小导致0文本块；后增预算、限制图谱、术语改写、真实重排并重新评估。实体关系校正后知识库存于lightrag-v3；旧版本留作过程快照。\n\n'
    text+='各次输出和时间按真实运行保留；模型服务并发、驻留和队列会影响耗时，不用单独检索或缓存时间冒充完整生成。部分用例不满足3秒，长期稳定性、真人语音及生产高并发未做完整验收。\n\n'
    (folder/'优化/过程问题记录.md').write_text(text,encoding='utf-8')
    (folder/'部署/复测.ps1').write_text("# 工单编号：人工智能NLP-RAG\nparam([string]$Python='D:\\an\\envs\\rags_\\python.exe')\nSet-Location -LiteralPath (Join-Path $PSScriptRoot '..')\n$env:PYTHONIOENCODING='utf-8'\n& $Python 测试\\回归测试.py\nif($LASTEXITCODE -ne 0){throw '回归失败'}\n& $Python 测试\\评估.py\n",encoding='utf-8-sig')
print('special metrics and process records saved')
