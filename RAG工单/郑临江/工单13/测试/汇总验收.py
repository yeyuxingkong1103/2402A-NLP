"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。区分交付与全部业务门槛通过。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
rows=['# 全部工单交付与验收状态','',
      '主文件夹仅保存工单1至工单13和用户原图片；每个工单均有独立源数据、索引、模型、代码、配置、测试与视频。',
      '功能实现/材料交付不等于全部指标已通过。以下保留实际结果；关键词覆盖、证据页命中不是问答准确率。','',
      '|工单|独立数据/五阶段/代码/视频|自动回归|指定题实测|完整3秒目标|专项说明|','|---|---|---|---|---|---|']
for num in range(1,14):
    f=ROOT/f'工单{num}'
    if num==1:
        d=json.loads((f/'测试/实测评估结果.json').read_text(encoding='utf-8'));answers=[r['rag'] for r in d['answers']];count=len(answers)
    else:
        d=json.loads((f/'测试/实测结果.json').read_text(encoding='utf-8'));answers=[r['result'] for r in d['answers']];count=len(answers)
    passed=all(r['generation']=='llm' and r['elapsed_ms']<=3000 for r in answers)
    if num==12:
        light=json.loads((f/'测试/LightRAG生成对比.json').read_text(encoding='utf-8'))
        passed=passed and all(row['result']['generation']=='llm' and row['result']['elapsed_ms']<=3000 for row in light)
    note=''
    if num==9:note='真实RAGAS精度/召回见指标汇总；单题及响应时间仍有未达标项'
    if num==10:note='Docker实际运行、模型网络与重启持久化通过；容器采用稀疏后端'
    if num==11:note='实际训练权重已交付，未见8题Recall@1 75%->87.5%'
    if num==12:note='官方LightRAG真实入库/检索/生成、RAGAS对比；稠密+图谱重排已修复实体混用'
    if num==13:note='性能专项平均2366.78ms->1015.57ms；期间有其他负载，非纯因果测量'
    log=(f/'测试/自动化测试日志.txt').read_text(encoding='utf-8',errors='replace') if (f/'测试/自动化测试日志.txt').exists() else ''
    rows.append(f"|{num}|已保存|{'9项通过' if 'OK' in log and 'FAILED (' not in log else '见原日志'}|{count}题|{'当前样本通过' if passed else '未全部通过'}|{note}|")
rows += ['','共同未完成的验收范围：生产高并发、24小时长时稳定性、浏览器真人语音。图谱采用有来源的规则型抽取，不推断未披露关系。',
         '工单8原eval_question.md附件缺失，已补建并标注；扫描空白提取页和年报表格抽取范围均如实记录。',
         '工单2-6的90%正确率及95%召回不能仅由关键词/Hit@5证明；逐题原始答案应配合人工事实检查，未标注“全量业务门槛全部通过”。']
(ROOT/'工单13/测试/全部工单验收汇总.md').write_text('\n'.join(rows),encoding='utf-8')
for num in range(2,14):
    p=ROOT/f'工单{num}/README.md'
    text=p.read_text(encoding='utf-8')
    extra='\n## 验收范围说明\n功能、材料和短时回归已交付；仍有部分指定问题完整耗时超过3秒。检索命中和关键词指标不是90%问答准确率或95%全事实召回率的证明。生产高并发、24小时稳定性及真人语音未全部验证；全部原始结果见本项目测试目录。\n'
    if '## 验收范围说明' not in text:p.write_text(text+extra,encoding='utf-8')
print('验收状态汇总保存到工单13内部')
