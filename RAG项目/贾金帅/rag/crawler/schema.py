# -*- coding: utf-8 -*-
"""
统一文档契约 + 路径集中管理 + 落盘工具。

铁律：爬虫只抓不理解——evidence_level / pico 一律留空，下游回填。
REST 系三类（guideline / chictr / wanfang）产出的 JSON 形状由本文件唯一决定，
下游 tools/ 直接读取，禁止在别处改字段。
"""
import json
import os

from .base import now_iso

# 必填字段（缺任一即视为失败，不落盘）
REQUIRED = ["doc_id", "source_type", "source_url", "title", "sections", "text", "crawl_time"]

# ======================= 路径集中管理（唯一出处） =======================
# 所有 data/raw 下的产物目录都在这里定义，别处禁止再硬编码路径字符串。
RAW = os.path.join("data", "raw")

PATHS = {
    # 源名: 产物目录（REST 系每个 doc 一个 JSON 落在这里）
    "guideline":   os.path.join(RAW, "guidelines"),
    "chictr":      os.path.join(RAW, "trials_cn"),
    "wanfang":     os.path.join(RAW, "literature_cn"),
    # otc_ann：每篇公告一个子目录（{ann_id}/ 下存原始附件 + index.json）
    "otc_ann":     os.path.join(RAW, "otc_ann"),
    # nmpa：进度 jsonl + 合并后的 drugs.json（不在子目录，保持一致习惯）
    "nmpa_progress": os.path.join(RAW, "drugs.jsonl"),
    "nmpa_out":      os.path.join(RAW, "drugs.json"),
    # nmpa 失败清单：抓不到任何批准文号记录的药名记在这里（曾经会被当成空记录落盘）
    "nmpa_failed":   os.path.join(RAW, "nmpa_failed.jsonl"),
    "yibao_file":    os.path.join(RAW, "yibao_catalog.pdf"),
    # 日志文件（唯一不在 data/raw 下内容目录里的条目）
    "crawler_log":   os.path.join(RAW, "crawler.log"),
    # 一把梭种子：本地全量药名清单（来自 KG 项目的药品数据），drug_name 字段
    "seed_names":   os.path.join(RAW, "drugs_standard.json"),
}


# ======================= 文档构造 / 校验 =======================
def build_doc(source_type, native_id, source_url, title, publish_date="",
              sections=None, text="", keywords=None, doc_id=None):
    """按统一 schema 构造文档。evidence_level / pico 恒留空（契约铁律）。

    doc_id 默认 `{source_type}:{native_id}`；需要短前缀（如 trials 的 `ct:`）
    或纯 id（guideline 的 `2024高血压指南`）时显式传入覆盖。
    """
    return {
        "doc_id": doc_id if doc_id else "%s:%s" % (source_type, native_id),
        "source_type": source_type,
        "source_url": source_url,
        "title": title,
        "publish_date": publish_date,
        "evidence_level": "",
        "pico": {"P": "", "I": "", "C": "", "O": ""},
        "sections": sections or {},
        "text": text,
        "keywords": keywords or [],
        "crawl_time": now_iso(),
    }


def safe_id(doc_id):
    """doc_id -> 文件名（冒号不能进 Windows 文件名）：chictr:123 -> chictr_123"""
    return doc_id.replace(":", "_")


def doc_path(out_dir, doc_id):
    """落盘路径：chictr:123 -> {out_dir}/chictr_123.json"""
    return os.path.join(out_dir, safe_id(doc_id) + ".json")


def done_ids(out_dir):
    """已落盘的 doc_id 集合（文件名形态，与 safe_id(doc_id) 可直接比对）。

    返回「文件名形态」而非还原回原始 doc_id —— 因为 doc_id 本身可能含下划线
    （如 guideline 的 2024_高血压指南），反推会歧义。调用方统一用 safe_id() 比对。
    """
    if not os.path.isdir(out_dir):
        return set()
    return {fn[:-5] for fn in os.listdir(out_dir)
            if fn.endswith(".json") and fn != "_failed.jsonl"}


def validate(doc):
    """返回缺失的必填字段名列表；全齐返回 []"""
    return [k for k in REQUIRED if not doc.get(k)]


# ======================= 落盘 / 失败记录 =======================
def save_json(path, obj):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def record_failure(out_dir, doc_id, reason):
    """失败记录追加到 {out_dir}/_failed.jsonl（断点续爬 / 报告共用）"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "_failed.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"doc_id": doc_id, "reason": reason}, ensure_ascii=False) + "\n")


def prune_failures(out_dir):
    """清掉已成功落盘（文件存在）的失败记录并按 doc_id 去重，保持与事实一致"""
    path = os.path.join(out_dir, "_failed.jsonl")
    if not os.path.exists(path):
        return
    seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not os.path.exists(doc_path(out_dir, rec["doc_id"])):
                seen[rec["doc_id"]] = line  # 同名保留最后一条
    with open(path, "w", encoding="utf-8") as f:
        for line in seen.values():
            f.write(line + "\n")
