# -*- coding: utf-8 -*-
# document_quality_assessment/pdf_page_analyze.py
from pypdf import PdfReader
from dataclasses import dataclass

@dataclass
class PdfPageResult:
    page_no:int
    char_count:int
    is_scan_page:bool

@dataclass
class PdfDocAnalyzeResult:
    file_path:str
    total_pages:int
    scan_page_cnt:int
    scan_ratio:float
    doc_type:str # scan / text / mix / error

def analyze_one_pdf(file_path:str, scan_char_threshold:int=150)->PdfDocAnalyzeResult:
    """
    PDF逐页分析；单页字符 < scan_char_threshold →扫描页
    scan_ratio>0.7 →扫描文档
    """
    try:
        reader = PdfReader(file_path)
        total = len(reader.pages)
        scan_cnt = 0
        for idx,page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            c = len(txt.strip())
            is_scan = bool(c < scan_char_threshold)
            if is_scan:
                scan_cnt +=1
        ratio = scan_cnt / total if total>0 else 0.0
        if ratio >=0.70:
            dtype = "scan"
        elif ratio <=0.10:
            dtype = "text"
        else:
            dtype = "mix"
        return PdfDocAnalyzeResult(
            file_path=file_path,
            total_pages=total,
            scan_page_cnt=scan_cnt,
            scan_ratio=round(ratio,3),
            doc_type=dtype
        )
    except Exception as e:
        return PdfDocAnalyzeResult(file_path=file_path,total_pages=0,scan_page_cnt=0,scan_ratio=0,doc_type="error")
