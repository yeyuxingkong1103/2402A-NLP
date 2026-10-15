# -*- coding: utf-8 -*-
# document_quality_assessment/quality_skill.py
import os
from .pdf_page_analyze import analyze_one_pdf
from .duplicate_detect import detect_duplicate
from .sensitive_detect import scan_file_sensitive
from .report_render import build_quality_report

class DocumentQualityAssessmentSkill:
    def __init__(self,config=None):
        self.cfg = config or {
            "pdf_scan_char_threshold":150,
            "pdf_scan_ratio_threshold":0.70,
            "simhash_sim_threshold":0.85,
            "enable_bankcard_detect":False
        }

    def run(self,doc_dir:str):
        """执行文档质量评估，返回完整quality_report"""
        all_file_paths = []
        format_stat = {}
        pdf_analyse_results = []
        doc_len_list = []
        for root,_,files in os.walk(doc_dir):
            for fname in files:
                fp = os.path.join(root,fname)
                ext = os.path.splitext(fname)[‑1].lower()
                format_stat[ext] = format_stat.get(ext,0)+1
                all_file_paths.append(fp)
                #简易文档长度统计
                try:
                    with open(fp,"r",encoding="utf‑8",errors="ignore") as f:
                        txt = f.read()
                        doc_len_list.append(len(txt))
                except:
                    doc_len_list.append(0)
                #pdf分析
                if ext==".pdf":
                    res = analyze_one_pdf(fp,self.cfg["pdf_scan_char_threshold"])
                    pdf_analyse_results.append(res)

        #重复检测
        exact_dup, sim_confirm = detect_duplicate(all_file_paths,self.cfg["simhash_sim_threshold"])
        #敏感检测
        sensitive_all = []
        for fpath in all_file_paths:
            s_list = scan_file_sensitive(fpath,self.cfg["enable_bankcard_detect"])
            sensitive_all.extend(s_list)

        #文档标签判定
        doc_label_map = {}
        for fp in all_file_paths:
            ext = os.path.splitext(fp)[‑1].lower()
            label = "Parse_Failed"
            if ext==".md":
                label = "Clean_Markdown"
            if ext==".pdf":
                for r in pdf_analyse_results:
                    if r.file_path == fp:
                        if r.doc_type=="scan":
                            label = "Scan_PDF"
                        else:
                            label = "Clean_Markdown"
            doc_label_map[fp] = label

        report = build_quality_report(
            format_stat=format_stat,
            len_list=doc_len_list,
            pdf_analyse_results=pdf_analyse_results,
            exact_dup_list=exact_dup,
            sim_confirm_list=sim_confirm,
            sensitive_audit_list=sensitive_all,
            doc_label_map=doc_label_map
        )
        return {"quality_report":report}
