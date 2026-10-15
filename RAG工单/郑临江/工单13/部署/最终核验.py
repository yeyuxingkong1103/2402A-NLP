"""工单编号：人工智能NLP-RAG项目-RAG性能瓶颈识别与优化。目录、来源、代码、文档、视频及排版核验。"""
import hashlib
import json
from pathlib import Path
import pymupdf
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]
all_records=[]
for num in range(1,14):
    folder=ROOT/f'工单{num}'
    preview=folder/'测试/排版预览';preview.mkdir(parents=True,exist_ok=True)
    document=pymupdf.open(folder/'五阶段完成方案.pdf')
    thumbs=[];overflow=[]
    for i,page in enumerate(document):
        for block in page.get_text('blocks'):
            if block[0]<0 or block[2]>page.rect.width+1 or block[3]>page.rect.height+1:overflow.append(i+1)
        pix=page.get_pixmap(matrix=pymupdf.Matrix(.8,.8));path=preview/f'page-{i+1:02}.png';pix.save(path)
        image=Image.open(path).convert('RGB');image.thumbnail((240,345));tile=Image.new('RGB',(260,380),'#e4eaec');tile.paste(image,((260-image.width)//2,5));ImageDraw.Draw(tile).text((10,360),f'WORKORDER {num} PAGE {i+1}',fill='black');thumbs.append(tile)
    for start in range(0,len(thumbs),8):
        contact=Image.new('RGB',(1040,760),'white')
        for j,tile in enumerate(thumbs[start:start+8]):contact.paste(tile,((j%4)*260,(j//4)*380))
        contact.save(preview/f'contact-{start//8+1}.png')
    sources=[]
    for pdf in (folder/'研发/data/documents').glob('*.pdf'):
        sources.append({'file':str(pdf.relative_to(folder)),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'bytes':pdf.stat().st_size})
    failures=[]
    for stage in ['设计','研发','测试','优化','部署']:
        if not (folder/stage).is_dir():failures.append('缺少阶段 '+stage)
    if overflow:failures.append('PDF溢出 '+str(overflow))
    if not (folder/'一键启动.cmd').exists():failures.append('缺少一键启动')
    if not (folder/'部署/完整演示视频.mp4').exists():failures.append('缺少演示')
    testname='实测评估结果.json' if num==1 else '实测结果.json'
    if not (folder/'测试'/testname).exists():failures.append('缺少实测')
    if num>1:
        for file in (folder/'研发/app').glob('*.py'):
            compile(file.read_text(encoding='utf-8'),str(file),'exec')
    record={'project':num,'pdf_pages':len(document),'layout_overflow':overflow,'independent_source_files':sources,'failures':failures,
        'video_bytes':(folder/'部署/完整演示视频.mp4').stat().st_size if (folder/'部署/完整演示视频.mp4').exists() else 0,
        'scope':'交付结构/语法/来源一致性；不把它当作全部业务验收合格'}
    (folder/'测试/最终交付核验.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');all_records.append(record)
    print(num,'PDF',len(document),'failures',failures,flush=True)
allowed={f'工单{i}' for i in range(1,14)}|{'9A5B8BBBA8E7E9657A7FF34CD2A65E20.png'}
outside=[p.name for p in ROOT.iterdir() if p.name not in allowed]
(ROOT/'工单13/测试/全部工单目录核验.json').write_text(json.dumps({'projects':all_records,'unexpected_root_entries':outside,'no_common_data_at_root':not outside},ensure_ascii=False,indent=2),encoding='utf-8')
assert not outside,outside
assert all(not r['failures'] for r in all_records),all_records
