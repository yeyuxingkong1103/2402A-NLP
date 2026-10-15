"""工单编号：人工智能NLP-RAG-基于PDF 文档的问答系统。交付文件核对及PDF视觉预览。"""
import json
from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
preview = BASE / "测试/临时预览/pdfs"
preview.mkdir(parents=True, exist_ok=True)
pdf = pymupdf.open(BASE / "五阶段完成方案.pdf")
thumbnails = []
overflow = []
for index, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2))
    path = preview / f"page-{index+1:02}.png"
    pix.save(path)
    im = Image.open(path).convert("RGB")
    im.thumbnail((250, 360))
    tile = Image.new("RGB", (270, 390), "#e5eaec")
    tile.paste(im, ((270-im.width)//2, 8))
    ImageDraw.Draw(tile).text((10,370),f"PAGE {index+1}",fill="black")
    thumbnails.append(tile)
    for block in page.get_text("blocks"):
        if block[0] < 0 or block[2] > page.rect.width + 1 or block[3] > page.rect.height + 1:
            overflow.append({"page":index+1,"bbox":block[:4]})
for start in range(0,len(thumbnails),12):
    canvas=Image.new("RGB",(1080,1170),"white")
    for n,tile in enumerate(thumbnails[start:start+12]):
        canvas.paste(tile,((n%4)*270,(n//4)*390))
    canvas.save(preview/f"contact-{start//12+1}.png")
assert not overflow, overflow
assert all((BASE / folder).is_dir() for folder in ['设计','研发','测试','优化','部署'])
assert (BASE / "一键启动.cmd").exists()
assert (BASE / "部署/完整演示视频.mp4").exists()
report=json.loads((BASE/"测试/实测评估结果.json").read_text(encoding="utf-8"))
assert len(report['answers'])==10
assert all(a['status']=='completed' for a in report['answers'])
result={"pages":len(pdf),"page_overflow":overflow,"stages":5,"comparison_questions":10,
        "video_bytes":(BASE/"部署/完整演示视频.mp4").stat().st_size}
(BASE/"测试/交付核验记录.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
