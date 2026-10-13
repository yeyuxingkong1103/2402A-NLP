from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"共享"))
from rag_workbench.core import chunk_text,read_text

def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",type=Path,default=Path("parse_report.json"));a=p.parse_args(); raw=read_text(a.input); clean=re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]","",raw); pages=raw.split("\f"); chunks=chunk_text(clean,a.input); report={"input":a.input,"characters_raw":len(raw),"characters_clean":len(clean),"pages":len(pages),"empty_pages":sum(not x.strip() for x in pages),"line_count":raw.count("\n")+bool(raw),"replacement_chars":raw.count("�"),"chunks":len(chunks),"needs_ocr":not bool(raw.strip())};a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
