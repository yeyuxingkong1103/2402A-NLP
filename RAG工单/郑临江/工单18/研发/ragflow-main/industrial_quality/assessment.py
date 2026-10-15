"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。
只读源资料；全页分类、MD5精确去重、SimHash近重复及敏感信息待审，不自动删除。
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import html
import json
import math
import re
import time
import zipfile
from pathlib import Path
import pymupdf
import yaml

DEFAULT=Path(__file__).with_name('assessment_config.yaml')

def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def simhash(text):
    terms=re.findall(r'[a-z0-9]+|[\u4e00-\u9fff]{1,}',text.lower())
    grams=[g for term in terms for g in ([term] if len(term)<4 else [term[i:i+3] for i in range(len(term)-2)])]
    weights=[0]*64
    for token,freq in collections.Counter(grams).items():
        code=int.from_bytes(hashlib.blake2b(token.encode(),digest_size=8).digest(),'big')
        for bit in range(64):weights[bit]+=freq if (code>>bit)&1 else -freq
    return sum(1<<bit for bit,w in enumerate(weights) if w>0)

def identity_valid(number):
    if not re.fullmatch(r'\d{17}[\dXx]',number):return False
    weights=[7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2]
    return '10X98765432'[sum(int(d)*w for d,w in zip(number[:17],weights))%11]==number[-1].upper()

def sensitive(text,cfg):
    patterns={'phone':r'(?<!\d)1[3-9]\d{9}(?!\d)','email':r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
              'identity_card':r'(?<!\d)\d{17}[0-9Xx](?!\d)',
              'bank_card':rf"(?<!\d)\d{{{cfg['sensitive']['bank_card_min_digits']},{cfg['sensitive']['bank_card_max_digits']}}}(?!\d)"}
    matches=[];ctx=cfg['context_characters']
    for kind,pattern in patterns.items():
        if not cfg['sensitive'].get(kind):continue
        for match in re.finditer(pattern,text):
            if kind=='identity_card' and not identity_valid(match.group()):continue
            matches.append({'type':kind,'value':match.group(),'offset':match.start(),
                'context':text[max(0,match.start()-ctx):match.end()+ctx],
                'status':'needs_human_review','note':'检测模式匹配不等于确认个人敏感信息；专利公开号等可能误报'})
            if len(matches)>=cfg['max_sensitive_matches_per_file']:return matches
    return matches

def inspect_file(path,cfg):
    path=Path(path);data=path.read_bytes();record={'path':str(path.resolve()),'name':path.name,'format':path.suffix.lower() or '[none]',
        'bytes':len(data),'md5':hashlib.md5(data).hexdigest(),'sha256':hashlib.sha256(data).hexdigest(),'labels':[],
        'pages':0,'page_characters':[],'scanned_pages':[],'review_reasons':[]}
    text=''
    try:
        if path.suffix.lower()=='.pdf':
            with pymupdf.open(path) as document:
                if document.needs_pass:raise ValueError('encrypted_pdf')
                record['pages']=len(document)
                texts=[page.get_text() for page in document]
                record['page_characters']=[len(re.sub(r'\s','',s)) for s in texts]
                record['scanned_pages']=[i+1 for i,n in enumerate(record['page_characters']) if n<cfg['page_character_threshold']]
                ratio=len(record['scanned_pages'])/max(1,len(document));record['scan_ratio']=ratio
                label='Scan_PDF' if ratio>cfg['scan_ratio_threshold'] else ('Text_PDF' if 1-ratio>=cfg['text_ratio_threshold'] else 'Mixed_PDF')
                record['labels'].append(label)
                record['image_pages']=[i+1 for i,p in enumerate(document) if p.get_images()]
                if min(abs(ratio-cfg['scan_ratio_threshold']),abs(ratio-(1-cfg['text_ratio_threshold'])))<=cfg['classification_review_margin'] or not len(document):record['review_reasons'].append('接近文字/扫描阈值或无页面')
                if record['scanned_pages'] and not record['image_pages']:record['review_reasons'].append('低文字页无嵌入位图，可能是矢量图/空白而非扫描')
                if label=='Scan_PDF':record['review_reasons'].append('低文本页也可能是图纸/空白，需OCR或抽样确认')
                record['route']='OCR' if label=='Scan_PDF' else ('PAGE_LEVEL' if label=='Mixed_PDF' else 'TEXT')
                text='\n'.join(texts)
        elif path.suffix.lower()=='.docx':
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(path) as z:
                root=ET.fromstring(z.read('word/document.xml'));text='\n'.join(node.text or '' for node in root.iter() if node.tag.endswith('}t'))
            record['labels'].append('Word_Document');record['route']='TEXT'
        elif path.suffix.lower() in {'.md','.txt','.csv','.json','.jsonl','.html','.xml'}:
            text=data.decode('utf-8-sig',errors='replace');record['labels'].append('Text_Document');record['route']='TEXT'
        else:record['route']='REVIEW';record['review_reasons'].append('不支持提取文本的格式')
        for label,terms in cfg['classification']['title_keywords'].items():
            if any(t.lower() in path.name.lower() for t in terms):record['labels'].append(label)
    except Exception as exc:
        record['error']=f'{type(exc).__name__}: {exc}';record['route']='REVIEW';record['labels'].append('Invalid_Document');record['review_reasons'].append('读取失败')
    record['characters']=len(text)
    record['length_measure']='embedded_text_only;扫描页未执行OCR，0字符不是空文档'
    record['sensitive_matches']=sensitive(text,cfg)
    record['simhash']=hex(simhash(text[:cfg['simhash_sample_chars']])) if len(text)>=cfg['simhash_min_chars'] else None
    record['text_preview']=text[:500]
    return record

def assess(paths,config=None,output_dir=None):
    cfg=yaml.safe_load(Path(config or DEFAULT).read_text(encoding='utf-8')) if not isinstance(config,dict) else config
    paths=[Path(p) for p in paths]
    output=Path(output_dir) if output_dir else None
    files=sorted({f.resolve() for p in paths for f in (p.rglob('*') if p.is_dir() else [p]) if f.is_file()
                  and (output is None or not f.resolve().is_relative_to(output.resolve()))})
    if output:output.mkdir(parents=True,exist_ok=True)
    cachefile=output/'inspection_cache.json' if output else None
    cache=json.loads(cachefile.read_text(encoding='utf-8')) if cachefile and cachefile.exists() else {}
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    records=[];start=time.perf_counter()
    for i,path in enumerate(files,1):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();key=digest+config_hash+path.name
        if key in cache:
            row={**cache[key],'path':str(path),'name':path.name}
        else:row=inspect_file(path,cfg);cache[key]=row
        records.append(row)
        if i%50==0 or i==len(files):
            print(f'质检进度 {i}/{len(files)}',flush=True)
            if output:atomic(cachefile,cache);atomic(output/'progress.json',{'processed':i,'total':len(files),'elapsed_s':time.perf_counter()-start})
    formats=collections.Counter(r['format'] for r in records);labels=collections.Counter(t for r in records for t in r['labels'])
    exact=collections.defaultdict(list)
    for r in records:exact[r['md5']].append(r['path'])
    duplicates=[{'md5':digest,'paths':names} for digest,names in exact.items() if len(names)>1]
    conflicts=[];seen=[]
    for row in records:
        if row['simhash'] is None:continue
        value=int(row['simhash'],16)
        for previous,other in seen:
            distance=(value^other).bit_count()
            if distance<=cfg['simhash_max_hamming'] and previous['md5']!=row['md5']:
                conflicts.append({'paths':[previous['path'],row['path']],'hamming_distance':distance,
                    'previews':[previous['text_preview'],row['text_preview']],'status':'needs_confirmation'})
        seen.append((row,value))
    lengths=sorted(r['characters'] for r in records)
    def percentile(p):
        if not lengths:return None
        position=(len(lengths)-1)*p;lo=math.floor(position);hi=math.ceil(position)
        return lengths[lo]+(lengths[hi]-lengths[lo])*(position-lo)
    bins=cfg['length_bins'];distribution=[]
    for i,low in enumerate(bins):
        high=bins[i+1] if i+1<len(bins) else None
        distribution.append({'min':low,'max_exclusive':high,'count':sum(n>=low and (high is None or n<high) for n in lengths)})
    report={'total_files':len(records),'format_distribution':{ext:{'count':n,'ratio':n/max(1,len(records))} for ext,n in formats.items()},
        'label_counts':dict(labels),'length_quantiles':{f'P{p}':percentile(p/100) for p in [25,50,75,90,99]},
        'length_distribution':distribution,'exact_duplicates':duplicates,'version_conflicts':conflicts,
        'pending_confirmation':[{'path':r['path'],'labels':r['labels'],'reasons':r['review_reasons'],'scan_ratio':r.get('scan_ratio')} for r in records if r['review_reasons']],
        'sensitive_review':[{'path':r['path'],**m} for r in records for m in r['sensitive_matches']],
        'files':records,'config':cfg,'elapsed_s':time.perf_counter()-start,
        'scope':'全页内嵌文本质检；图像中的敏感信息尚需OCR后复检，不声称已读取全部扫描文字'}
    if output:
        atomic(output/'quality_report.json',report)
        sections=['<h1>文档质量评估报告</h1>',f'<p>文件 {len(records)} | 处理 {report["elapsed_s"]:.2f}s</p>']
        for name in ['format_distribution','label_counts','length_quantiles','length_distribution','exact_duplicates','version_conflicts','pending_confirmation','sensitive_review']:
            sections.append('<h2>'+name+'</h2><pre>'+html.escape(json.dumps(report[name],ensure_ascii=False,indent=2))+'</pre>')
        (output/'quality_report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px/1.7 Microsoft YaHei;margin:40px;max-width:1200px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f6;padding:20px}</style>'+''.join(sections),encoding='utf-8')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',nargs='+',required=True);parser.add_argument('--output',required=True);parser.add_argument('--config',default=str(DEFAULT));args=parser.parse_args()
    report=assess(args.input,args.config,args.output);print(json.dumps({k:report[k] for k in ['total_files','label_counts','length_quantiles','elapsed_s']},ensure_ascii=False))
