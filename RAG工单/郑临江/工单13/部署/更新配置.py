"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。独立源码维护。"""
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for num in range(2,14):
    folder=ROOT/f'工单{num}'
    if num!=13:shutil.copy2(ROOT/'工单13/研发/app/core.py',folder/'研发/app/core.py')
    base=folder/'研发/app/base.py'
    text=base.read_text(encoding='utf-8').replace('先给简洁答案，再给页码，不编造事实。','以120字左右给出所问事实和页码，省去背景与重复解释，不编造事实。')
    text=text.replace('"max_tokens": 500','"max_tokens": 220').replace('"num_predict": 500','"num_predict": 220')
    text=text.replace("concept_words = re.findall(r\"[\\u4e00-\\u9fff]{2,}\", understand(question)[\"retrieval_query\"])",
        "concept_words = list({s[i:i+2] for s in re.findall(r'[\\u4e00-\\u9fff]{2,}', understand(question)['retrieval_query']) for i in range(len(s)-1)})")
    text=text.replace("if h.get('kind', 'text') not in {'table', 'glossary'} and len(h['text']) > 900:",
        "if '年度报告' in h['document']:\n                    snippet = h['text'][:1800]\n                elif h.get('kind', 'text') not in {'table', 'glossary'} and len(h['text']) > 900:")
    text=text.replace("snippet = h['text'][:1800]","snippet = re.sub(r'(?m)^\\s*\\d+(?:/\\d+)?\\s*$', '', h['text'])[:1800]")
    base.write_text(text,encoding='utf-8')
    html=folder/'研发/static/index.html'
    text=html.read_text(encoding='utf-8')
    text=text.replace("let currentQuestion='',poller;","let currentQuestion='',currentSources=[],poller;")
    text=text.replace("currentQuestion=q;","currentQuestion=q;currentSources=r.sources;")
    text=text.replace("rating:b.dataset.rating}","rating:b.dataset.rating,source_id:currentSources[0]?.id}")
    html.write_text(text,encoding='utf-8')
print('本地副本及生成长度配置更新完成')
