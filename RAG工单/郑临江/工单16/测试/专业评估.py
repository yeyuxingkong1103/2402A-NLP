"""工单编号：人工智能NLP-RAG-微调专用视觉语言模型工单。MCQ指标与生成文本指标明确区分。"""
import json,math,collections
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];R=BASE/'研发';p=BASE/'测试/真实微调结果.json';r=json.loads(p.read_text(encoding='utf-8'))
lookup={row['question']:row for row in [json.loads(x) for x in (R/'data/finetune/test.jsonl').read_text(encoding='utf-8').splitlines()]}
def rouge_l(a,b):
    previous=[0]*(len(b)+1)
    for ca in a:
        current=[0]
        for j,cb in enumerate(b):current.append(previous[j]+1 if ca==cb else max(previous[j+1],current[-1]))
        previous=current
    return 2*previous[-1]/max(1,len(a)+len(b))
def bleu1(a,b):
    overlap=sum((collections.Counter(a)&collections.Counter(b)).values());precision=overlap/max(1,len(a))
    return precision*min(1,math.exp(1-len(b)/max(1,len(a))))
for phase in ['before','after']:
    samples=r[phase]['samples']
    for s in samples:
        ref=lookup[s['question']]['reference'];s['raw_generation_character_BLEU1']=bleu1(s['raw_output'],ref);s['raw_generation_character_ROUGE_L']=rouge_l(s['raw_output'],ref)
    r[phase]['mean_character_BLEU1']=sum(s['raw_generation_character_BLEU1'] for s in samples)/len(samples)
    r[phase]['mean_character_ROUGE_L']=sum(s['raw_generation_character_ROUGE_L'] for s in samples)/len(samples)
r['metric_definitions']={'terminology_accuracy':'group=1的专业文本选择题正确字母比例；不是开放术语生成准确率',
    'drawing_reasoning_accuracy':'group=2图纸题选择字母正确率','BLEU1':'原始生成与参考选项内容的字符级BLEU-1，非标准词级多阶BLEU',
    'ROUGE_L':'原始生成与参考选项内容的字符级LCS-F1；选项字母正确仍可能生成无关后缀，单独检查原文'}
p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'测试/评估结果.json').write_text(json.dumps({'status':'real_lora_training_completed','before':r['before'],'after':r['after'],'epochs':r['epochs'],'definitions':r['metric_definitions']},ensure_ascii=False,indent=2),encoding='utf-8')
print('actual training evaluation refreshed')
