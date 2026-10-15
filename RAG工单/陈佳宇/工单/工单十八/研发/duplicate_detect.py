# -*- coding: utf-8 -*-
# document_quality_assessment/duplicate_detect.py
import hashlib
import simhash

def calc_md5(file_path:str)->str:
    hash_md5 = hashlib.md5()
    with open(file_path,"rb") as f:
        for chunk in iter(lambda: f.read(4096),b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_text_snippet(file_path:str)->str:
    """简易获取文档文本用于simhash，扩展可对接真实文档解析"""
    try:
        with open(file_path,"r",encoding="utf‑8",errors="ignore") as f:
            return f.read(8000)
    except:
        return ""

def calc_simhash(text:str):
    tokens = text.replace("\n"," ").split()
    return simhash.Simhash(tokens).value

def detect_duplicate(file_list:list,sim_threshold=0.85):
    """
    返回:
        exact_dup:完全重复md5分组
        sim_need_confirm:近似待确认版本冲突列表 [(fileA,fileB,sim_score)]
    """
    md5_map = {}
    sim_list = []
    for fp in file_list:
        md5 = calc_md5(fp)
        if md5 not in md5_map:
            md5_map[md5] = []
        md5_map[md5].append(fp)
        snippet = get_file_text_snippet(fp)
        sh_val = calc_simhash(snippet)
        sim_list.append((fp,sh_val))

    exact_dup = [v for v in md5_map.values() if len(v)>=2]
    sim_need_confirm = []
    for i in range(len(sim_list)):
        f1,s1 = sim_list[i]
        for j in range(i+1,len(sim_list)):
            f2,s2 = sim_list[j]
            distance = simhash.num_bits‑simhash.Simhash._distance(s1,s2)
            score = distance / simhash.num_bits
            if score >= sim_threshold:
                sim_need_confirm.append({"file1":f1,"file2":f2,"similar":round(score,3)})
    return exact_dup, sim_need_confirm
