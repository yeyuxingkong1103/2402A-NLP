"""工单14-18：原始数据、模型、代码路径与渲染核验，不以目录齐全冒充全部业务验收。"""
import json,hashlib,zipfile,re
from pathlib import Path
from PIL import Image,ImageDraw
import pymupdf
ROOT=Path(__file__).resolve().parents[2]
source_zip=zipfile.ZipFile('D:/软件/QQ/data/RAG 工单1/14-17附件/original_problems.zip')
expected={Path(info.filename).name:info.file_size for info in source_zip.infolist() if info.filename.lower().endswith('.pdf')}
checks=[]
for n in range(14,19):
    f=ROOT/f'工单{n}';config=json.loads((f/'研发/project.json').read_text(encoding='utf-8'))
    docs=f/'研发/original_problems/documents';native=[docs/name for name in expected if (docs/name).exists()]
    assert len(native)==1700,(n,len(native))
    assert all(p.stat().st_size==expected[p.name] for p in native),(n,'来源文件大小不一致')
    assert (f/'研发/original_problems/questions.jsonl').exists()
    for stage in ['设计','研发','测试','优化','部署']:assert (f/stage).is_dir()
    assert (f/'一键启动.cmd').exists()
    pdf=pymupdf.open(f/'五阶段完成方案.pdf');preview=f/'测试/排版预览';preview.mkdir(exist_ok=True);tiles=[]
    for i,page in enumerate(pdf):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1,1));pix.save(preview/f'page-{i+1:02}.png')
        for block in page.get_text('blocks'):
            assert 0<=block[0]<page.rect.width and block[2]<=page.rect.width+1 and block[3]<=page.rect.height+1,(n,i,block)
        im=Image.open(preview/f'page-{i+1:02}.png').convert('RGB');im.thumbnail((240,345));tile=Image.new('RGB',(260,380),'#e5ecee');tile.paste(im,((260-im.width)//2,5));ImageDraw.Draw(tile).text((5,360),f'WORKORDER {n} PAGE {i+1}',fill='black');tiles.append(tile)
    for start in range(0,len(tiles),8):
        canvas=Image.new('RGB',(1040,760),'white')
        for j,tile in enumerate(tiles[start:start+8]):canvas.paste(tile,((j%4)*260,(j//4)*380))
        canvas.save(preview/f'contact-{start//8+1}.png')
    syntax=[]
    for folder in [f/'研发/app',f/'研发/document_quality_assessment']:
        for p in folder.glob('*.py'):
            compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');syntax.append(str(p.relative_to(f)))
    record={'project':n,'source_pdfs':len(native),'pdf_pages':len(pdf),'layout_verified_no_overflow':True,'syntax_checked':syntax,
            'runtime_data_root':'当前工单研发目录','formal_ragflow_status':json.loads((f/'部署/正式环境状态.json').read_text(encoding='utf-8'))['status'],
            'video_exists':(f/'部署/完整演示视频.mp4').exists()}
    (f/'测试/交付结构核验.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8');checks.append(record)
allowed={f'工单{n}' for n in range(1,19)}|{'9A5B8BBBA8E7E9657A7FF34CD2A65E20.png'}
unexpected=[p.name for p in ROOT.iterdir() if p.name not in allowed]
assert not unexpected,unexpected
(ROOT/'工单18/测试/本批目录隔离核验.json').write_text(json.dumps({'projects':checks,'unexpected_root_entries':unexpected,'common_data_at_root':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False))
