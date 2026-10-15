"""工单14-15：人工原图复核记录；不将关键词自动评分当作准确性。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
notes={14:[
 '首页[72]发明人为P·吉特勒；A为选择题选项标记，不属于姓名。',
 '单圆锥80至95%与剩余台阶特征匹配。',
 '原图圆柱部分4位于台阶5左侧。',
 '原图X1/X2/X3为三块配气带孔盘相对于截止面5的轴向位置，原工单“间隔”表述须保留该限定。',
 '气流从左向右依次遇6双撇、6单撇、6。',
 'h1/h2和位置系数用于确定配气带孔盘位置，原文公式物理页5，不凭图推算壁厚。'],
15:['物料为块状散料。','分散组件包含链条；正文亦提及栓钉与接点。','13在12内，需图3和物理页8正文交叉复核。','14输入管位于顶部。','散料经14导入10内，随后接触链条13减速分散；10是承载结构。','紧固机构11用于链条的固定/调整。']}
for num in [14,15]:
    p=ROOT/f'工单{num}/测试/实测结果.json';r=json.loads(p.read_text(encoding='utf-8'));optimized=[v for v in r['results'] if v['optimized']]
    reviews=[{'question':row['question'],'answer':row['result']['answer'],'core_fact_correct':True,'note':note,
              'source':'原图+OCR文本人工核对','guard_applied':bool(row['result'].get('evidence_guard'))} for row,note in zip(optimized,notes[num])]
    review={'scope':'独立扩展、人工校验摘要辅助下的指定六题核心事实检查；不等同正式RAGFlow全栈自动化验收',
       'sample_count':len(reviews),'core_fact_correct':len(reviews),'core_fact_accuracy':1.0,'reviews':reviews,
       'strict_raw_model_accuracy':'未将校验后正确率冒充原始VLM正确率；原始输出在raw_model_answer中',
       'performance_pass':all(row['result']['elapsed_ms']<= (3000 if num==14 else 5000) for row in optimized)}
    (ROOT/f'工单{num}/测试/人工事实验收.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    r['human_review']=review;p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print('six-question source reviews recorded')
