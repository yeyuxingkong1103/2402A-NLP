from __future__ import annotations
import argparse,json,re
from pathlib import Path

def pages(path):
    try:
        from pypdf import PdfReader
        return [(i+1,p.extract_text() or "") for i,p in enumerate(PdfReader(str(path),strict=False).pages)]
    except Exception as exc:
        return [(0,f"[parse-error] {type(exc).__name__}: {exc}")]
def repair(text):
    text=text.replace("\u00ad","").replace("\ufffd","")
    text=re.sub(r"(?<=\w)-\n(?=\w)","",text)
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()
def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",type=Path,default=Path("repaired.json"));a=p.parse_args();rows=[]
    for no,text in pages(Path(a.input)):
        fixed=repair(text);rows.append({"page":no,"raw_characters":len(text),"repaired_characters":len(fixed),"empty":not bool(fixed),"suspected_ocr":len(text.strip())<30,"text":fixed})
    r={"input":a.input,"pages":rows,"summary":{"pages":len(rows),"empty_pages":sum(x["empty"] for x in rows),"suspected_ocr_pages":sum(x["suspected_ocr"] for x in rows)}};a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(r["summary"],ensure_ascii=False))
if __name__=="__main__":main()
