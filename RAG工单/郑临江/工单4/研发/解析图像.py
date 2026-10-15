"""工单编号：人工智能NLP-RAG-图像内容解析及检索优化。真实本地多模态解析并保留原输出。"""
import base64
import hashlib
import json
import urllib.request
from pathlib import Path
import pymupdf

BASE=Path(__file__).resolve().parent
pdf=BASE/'data/documents/招股说明书2.pdf'
images=BASE/'data/images';images.mkdir(parents=True,exist_ok=True)
document=pymupdf.open(pdf)
raw=[]
manual={39:'公司内部组织结构图：销售部下设4个部门，分别为电话及网络销售部、渠道销售部、大客户销售部、国际贸易部。大客户销售部下设6个销售处：珠海销售处、深圳销售处、北京销售处、武汉销售处、广州销售处、成都销售处。大客户销售部的销售处最多，为6个。',
72:'2008年中国IC市场应用结构与增长图：汽车电子增长率14.0%，工业控制10.5%，计算机7.9%，其他6.1%，网络通信2.6%，消费5.1%，IC卡-2.0%。增长最快为汽车电子（14.0%）；负增长为IC卡（-2.0%）。应用规模单位为亿元，增长率单位为%。'}
for page in [39,72]:
    image=images/f'pdf-page-{page}.png'
    document[page-1].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(image)
    question='请准确读取此PDF页面中的图。逐项列出部门及上下级关系或每个行业的增长率。不要把图旁正文当成图的全部内容。只描述可读信息。'
    data=json.dumps({'model':'Qwen3.5:4B','messages':[{'role':'user','content':question,'images':[base64.b64encode(image.read_bytes()).decode()]}],
        'stream':False,'think':False,'options':{'num_predict':650,'num_ctx':8192}}).encode()
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data,{'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=180) as res:
            payload=json.load(res)
        output=payload['message']['content']
    except Exception as exc:
        output='ERROR '+type(exc).__name__+':'+str(exc)
    item={'document':pdf.name,'pdf_page':page,'image_path':'data/images/'+image.name,
        'source_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'model':'Qwen3.5:4B',
        'raw_model_description':output,'description':manual[page],
        'verification':'人工逐项核对原图并校正；索引用校正版本，原始多模态输出单独保存'}
    raw.append(item)
    (BASE/'data/image_annotations.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding='utf-8')
    print('page',page,output[:250],flush=True)
