"""工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'测试/gold.json').read_text(encoding='utf-8'))
report={'baseline':{'status':'not_run','reason':'完整视觉生成模型基线未在本机启动，避免伪造BLEU/ROUGE'},'finetune':{'status':'prepared','dataset':str(ROOT/'研发/data'),'config':str(ROOT/'研发/data/lora_config.json'),'epochs_target':1},'industrial_metrics':{'terminology_accuracy':'待训练后实测','drawing_relation_accuracy':'待训练后实测','bleu':'未执行','rouge':'未执行'},'failure_analysis':['图像输入需按VLM chat template转换','当前只生成训练配置和JSONL，未冒称权重已训练']}
(ROOT/'测试/评估结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report)
