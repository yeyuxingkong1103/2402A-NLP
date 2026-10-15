"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流。分层抽样全页预览。"""
import json,random
from pathlib import Path
import pymupdf
from PIL import Image,ImageDraw
BASE=Path(__file__).resolve().parents[1];report=json.loads((BASE/'测试/全量质检/quality_report.json').read_text(encoding='utf-8'))
out=BASE/'测试/人工抽查';out.mkdir(exist_ok=True)
selected=[]
for label in ['Scan_PDF','Text_PDF','Mixed_PDF']:
    rows=[r for r in report['files'] if label in r['labels']];random.Random(42).shuffle(rows)
    selected+=rows[:3]
for index,row in enumerate(selected):
    thumbs=[]
    with pymupdf.open(row['path']) as doc:
        for n,page in enumerate(doc):
            pix=page.get_pixmap(matrix=pymupdf.Matrix(.6,.6));img=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);img.thumbnail((180,250))
            tile=Image.new('RGB',(200,285),'#e8eff0');tile.paste(img,((200-img.width)//2,5));ImageDraw.Draw(tile).text((7,260),f"P{n+1} chars:{row['page_characters'][n]}",fill='black');thumbs.append(tile)
    for start in range(0,len(thumbs),20):
        image=Image.new('RGB',(1000,285*((min(20,len(thumbs)-start)+4)//5)),'white')
        for i,tile in enumerate(thumbs[start:start+20]):image.paste(tile,((i%5)*200,(i//5)*285))
        image.save(out/f'{index+1:02}-{Path(row["name"]).stem}-{start//20}.png')
(out/'抽样来源.json').write_text(json.dumps([{'index':i+1,'name':r['name'],'labels':r['labels'],'pages':r['pages'],'scan_ratio':r['scan_ratio']} for i,r in enumerate(selected)],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{'index':i+1,'name':r['name'],'labels':r['labels'],'pages':r['pages'],'scan_ratio':r['scan_ratio']} for i,r in enumerate(selected)],ensure_ascii=False))
