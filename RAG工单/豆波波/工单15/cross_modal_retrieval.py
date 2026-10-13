from __future__ import annotations
import argparse,json,re
from pathlib import Path
def tokens(s):return set(re.findall(r"[A-Za-z0-9_./-]+|[\u4e00-\u9fff]",s.lower()))
def inspect(path):
    text="";images=[]
    try:
        from pypdf import PdfReader
        reader=PdfReader(str(path),strict=False);text="\n".join(p.extract_text() or "" for p in reader.pages)
        for page_no,page in enumerate(reader.pages,1):
            xobj=page.get("/Resources",{}).get_object().get("/XObject",{})
            for name,ref in xobj.items():
                obj=ref.get_object()
                if obj.get("/Subtype")=="/Image":images.append({"id":f"image-{page_no}-{name}","page":page_no,"name":str(name),"text":"","width":obj.get("/Width"),"height":obj.get("/Height")})
    except Exception as exc: return {"text":text,"images":images,"warning":f"{type(exc).__name__}: {exc}"}
    return {"text":text,"images":images}
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--query",required=True);p.add_argument("--output",type=Path,default=Path("cross_modal.json"));a=p.parse_args();data=inspect(Path(a.input));q=tokens(a.query);records=[{"modality":"text","score":len(q&tokens(data["text"]))/max(len(q),1),"content":data["text"][:2000]}]
 for img in data["images"]: records.append({"modality":"image","score":0.0,"content":img,"reason":"需要 OCR 或图像向量模型补充语义"})
 records.sort(key=lambda x:x["score"],reverse=True);r={"query":a.query,"results":records,"image_count":len(data["images"]),"warning":data.get("warning")};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({"image_count":len(data["images"]),"results":len(records)},ensure_ascii=False))
if __name__=="__main__":main()
