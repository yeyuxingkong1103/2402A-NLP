# -*- coding: utf-8 -*-
# document_quality_assessment/sensitive_detect.py
import re

PHONE_PAT = re.compile(r"1[3‑9]\d{9}")
EMAIL_PAT = re.compile(r"[a‑zA‑Z0‑9_\‑\.]+@[a‑zA‑Z0‑9‑]+\.[a‑zA‑Z\.]{2,}")
IDCARD_PAT = re.compile(r"[1‑9]\d{5}(18|19|20)\d{2}(0[1‑9]|1[0‑2])(0[1‑9]|[12]\d|3[01])\d{3}[\dXx]")
BANKCARD_PAT = re.compile(r"\d{16,19}")

def extract_context(text:str,pos:int,win=60):
    start = max(0,pos‑win)
    end = min(len(text),pos+win)
    return text[start:end].replace("\n"," ")

def scan_file_sensitive(file_path:str,enable_bankcard=False):
    """返回待审核列表，每条带上下文"""
    out = []
    try:
        with open(file_path,"r",encoding="utf‑8",errors="ignore") as f:
            full_txt = f.read()
    except:
        return out
    #手机号
    for m in PHONE_PAT.finditer(full_txt):
        out.append({"file":file_path,"type":"phone","match":m.group(),"context":extract_context(full_txt,m.start())})
    #邮箱
    for m in EMAIL_PAT.finditer(full_txt):
        out.append({"file":file_path,"type":"email","match":m.group(),"context":extract_context(full_txt,m.start())})
    #身份证
    for m in IDCARD_PAT.finditer(full_txt):
        out.append({"file":file_path,"type":"idcard","match":m.group(),"context":extract_context(full_txt,m.start())})
    #银行卡，默认关闭
    if enable_bankcard:
        for m in BANKCARD_PAT.finditer(full_txt):
            out.append({"file":file_path,"type":"bankcard","match":m.group(),"context":extract_context(full_txt,m.start())})
    return out
