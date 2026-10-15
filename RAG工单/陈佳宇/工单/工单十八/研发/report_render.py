# -*- coding: utf-8 -*-
# document_quality_assessment/report_render.py
import statistics

def calc_percentile(data_list, q):
    if not data_list:
        return 0
    s = sorted(data_list)
    idx = int(len(s)* q /100)
    return s[idx]

def build_quality_report(
    format_stat,
    len_list,
    pdf_analyse_results,
    exact_dup_list,
    sim_confirm_list,
    sensitive_audit_list,
    doc_label_map
):
    """组装完整结构化质检报告"""
    len_p25 = calc_percentile(len_list,25)
    len_p50 = calc_percentile(len_list,50)
    len_p75 = calc_percentile(len_list,75)
    len_p90 = calc_percentile(len_list,90)
    len_p99 = calc_percentile(len_list,99)

    report = {
        "global_stat":{
            "total_doc_count":sum(format_stat.values()),
            "format_distribution":format_stat,
            "char_length_distribution":{
                "P25":len_p25,
                "P50":len_p50,
                "P75":len_p75,
                "P90":len_p90,
                "P99":len_p99
            }
        },
        "pdf_scan_stats":[r.__dict__ for r in pdf_analyse_results],
        "exact_duplicate_groups":exact_dup_list,
        "need_confirm_sim_version_conflict":sim_confirm_list,
        "need_audit_sensitive":sensitive_audit_list,
        "document_label":doc_label_map
    }
    return report
