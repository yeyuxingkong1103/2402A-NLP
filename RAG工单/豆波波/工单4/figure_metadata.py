from __future__ import annotations
import argparse,json
from pathlib import Path

def extract(path):
    rows=[]
    try:
        from pypdf import PdfReader
        reader=PdfReader(str(path),strict=False)
        for page_no,page in enumerate(reader.pages,1):
            resources=page.get("/Resources",{}).get_object(); xobj=resources.get("/XObject",{}) if resources else {}
            for name,ref in xobj.items():
                obj=ref.get_object()
                if obj.get("/Subtype")=="/Image": rows.append({"page":page_no,"name":str(name),"width":obj.get("/Width"),"height":obj.get("/Height"),"color_space":str(obj.get("/ColorSpace","")),"ocr_text":"","embedding":None})
    except Exception as exc: return {"input":str(path),"images":[],"warning":f"{type(exc).__name__}: {exc}"}
    return {"input":str(path),"images":rows,"count":len(rows)}

def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",type=Path,default=Path("figures.json"));a=p.parse_args();r=extract(Path(a.input));a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r,ensure_ascii=False))
if __name__=="__main__":main()
