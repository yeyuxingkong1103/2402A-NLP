#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DocumentQualityAssessmentSkill - 主入口 (pdfplumber 版)"""
import os, sys, re, json, hashlib, time
import yaml
from collections import Counter, defaultdict
import pdfplumber
import numpy as np


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pdf_pages_text(path, max_pages=None):
    """用 pdfplumber 逐页返回文本列表"""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            n = len(pdf.pages)
            limit = min(n, max_pages) if max_pages else n
            for i in range(limit):
                try:
                    t = pdf.pages[i].extract_text() or ""
                except Exception:
                    t = ""
                pages.append(t)
    except Exception as e:
        raise e
    return pages


def format_stats(files):
    cnt = Counter()
    for f in files:
        cnt[os.path.splitext(f)[1].lower().lstrip(".") or "unknown"] += 1
    total = len(files)
    return {"total_files": total,
            "distribution": [{"ext": k, "count": v,
                              "ratio": round(v/total, 4) if total else 0}
                             for k, v in cnt.most_common()]}


def classify_pdf(path, cfg):
    r = {"path": path, "total_pages": 0, "scan_pages": 0, "text_pages": 0,
         "scan_ratio": 0.0, "type": "unknown", "has_error": False, "error": ""}
    try:
        max_pages = cfg.get("max_pages_per_file", 100)
        thresh = cfg.get("scan_page_char_threshold", 50)
        pages = _pdf_pages_text(path, max_pages)
        for t in pages:
            if len((t or "").strip()) < thresh:
                r["scan_pages"] += 1
            else:
                r["text_pages"] += 1
        r["total_pages"] = len(pages)
        if r["total_pages"] > 0:
            r["scan_ratio"] = round(r["scan_pages"]/r["total_pages"], 4)
        sr = r["scan_ratio"]
        if sr >= cfg.get("scan_ratio_threshold", 0.7):
            r["type"] = "scan"
        elif sr <= cfg.get("mixed_ratio_lower", 0.1):
            r["type"] = "text"
        else:
            r["type"] = "mixed"
    except Exception as e:
        r["has_error"] = True
        r["error"] = str(e)
        r["type"] = "parse_failed"
    return r


def classify_pdfs(files, cfg):
    pdfs = [f for f in files if f.lower().endswith(".pdf")]
    details = []
    for p in pdfs:
        details.append(classify_pdf(p, cfg))
    c = Counter(d["type"] for d in details)
    return {"summary": {"total_pdfs": len(pdfs),
                        "text_type": c.get("text", 0),
                        "scan_type": c.get("scan", 0),
                        "mixed_type": c.get("mixed", 0),
                        "parse_failed": c.get("parse_failed", 0),
                        "scan_ratio_overall": round(c.get("scan", 0)/len(pdfs), 4) if pdfs else 0},
            "details": details}


def get_length(path):
    try:
        if path.lower().endswith(".pdf"):
            pages = _pdf_pages_text(path)
            return sum(len(p) for p in pages)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return len(f.read())
    except Exception:
        return 0


def length_stats(files, cfg):
    lengths = [get_length(f) for f in files]
    lengths = [l for l in lengths if l > 0]
    if not lengths:
        return {"count": 0, "percentiles": {}, "bins": []}
    arr = np.array(lengths)
    pct = {}
    for p in cfg.get("percentiles", [25, 50, 75, 90, 99]):
        pct[f"P{p}"] = int(np.percentile(arr, p))
    edges = cfg.get("bins", [0, 1000, 5000, 10000, 50000, 100000])
    labels = cfg.get("bin_labels", [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)])
    bins = []
    for i in range(len(edges)-1):
        cnt = int(((arr >= edges[i]) & (arr < edges[i+1])).sum())
        rng = labels[i] if i < len(labels) else f"{edges[i]}-{edges[i+1]}"
        bins.append({"range": rng, "count": cnt, "ratio": round(cnt/len(arr), 4)})
    return {"count": len(arr), "percentiles": pct, "mean": int(arr.mean()),
            "max": int(arr.max()), "min": int(arr.min()), "bins": bins}


def md5_file(path, chunk=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def extract_sample(path, n):
    try:
        if path.lower().endswith(".pdf"):
            pages = _pdf_pages_text(path)
            return "".join(pages)[:n]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(n)
    except Exception:
        return ""


def simhash(text, bits=64):
    if not text: return 0
    tokens = [text[i:i+3] for i in range(len(text)-2)] or [text]
    v = [0]*bits
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i in range(bits):
        if v[i] > 0: fp |= (1 << i)
    return fp


def hamming(a, b):
    return bin(a ^ b).count("1")


def dedup(files, cfg):
    md5_groups = defaultdict(list)
    for f in files:
        try:
            md5_groups[md5_file(f)].append(f)
        except Exception:
            pass
    exact = [{"md5": h, "files": ps} for h, ps in md5_groups.items() if len(ps) > 1]
    pending = []
    if cfg.get("simhash_enabled", True):
        exact_set = {p for g in exact for p in g["files"]}
        uniq = [f for f in files if f not in exact_set]
        sample_n = cfg.get("simhash_sample_size", 10000)
        thresh = cfg.get("simhash_hamming_threshold", 3)
        hashes = {}
        for f in uniq:
            t = extract_sample(f, sample_n)
            if t:
                hashes[f] = simhash(t)
        items = list(hashes.items())
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                d = hamming(items[i][1], items[j][1])
                if d <= thresh:
                    pending.append({"file_a": items[i][0], "file_b": items[j][0],
                                    "hamming_distance": d,
                                    "reason": "SimHash 相似度超过阈值，请人工确认"})
    return {"summary": {"total_files": len(files),
                        "exact_duplicate_groups": len(exact),
                        "exact_duplicate_files": sum(len(g["files"]) for g in exact),
                        "pending_conflict_pairs": len(pending),
                        "md5_enabled": cfg.get("md5_enabled", True),
                        "simhash_enabled": cfg.get("simhash_enabled", True)},
            "exact_duplicates": exact,
            "pending_conflicts": pending}


def extract_text(path, n=200000):
    try:
        if path.lower().endswith(".pdf"):
            pages = _pdf_pages_text(path)
            return "\n".join(pages)[:n]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(n)
    except Exception:
        return ""


def scan_sensitive(files, cfg):
    pending = []
    counters = {"phone": 0, "email": 0, "id_card": 0, "bank_card": 0}
    ctx = cfg.get("context_chars", 40)
    for f in files:
        text = extract_text(f)
        if not text:
            continue
        for key in ["phone", "email", "id_card", "bank_card"]:
            sub = cfg.get(key, {})
            if not sub.get("enabled", False):
                continue
            pat = sub.get("pattern")
            if not pat:
                continue
            try:
                regex = re.compile(pat)
            except re.error:
                continue
            for m in regex.finditer(text):
                s, e = m.start(), m.end()
                c_start = max(0, s-ctx); c_end = min(len(text), e+ctx)
                context = text[c_start:c_end].replace("\n", " ")
                pending.append({"file": f, "type": key, "match": m.group(),
                                "position": [s, e], "context": f"...{context}..."})
                counters[key] += 1
    summary = {"total_hits": sum(counters.values()),
               "phone_hits": counters["phone"],
               "email_hits": counters["email"],
               "id_card_hits": counters["id_card"],
               "bank_card_hits": counters["bank_card"],
               "bank_card_enabled": cfg.get("bank_card", {}).get("enabled", False)}
    return {"summary": summary, "pending_items": pending}


def classify_docs(files, pdf_info, dedup_result, cfg):
    pdf_map = {d["path"]: d for d in pdf_info.get("details", [])}
    dup_set = {p for g in dedup_result.get("exact_duplicates", []) for p in g["files"]}
    docs = []
    counter = Counter()
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        labels = []
        if f in dup_set:
            labels.append("Duplicate")
        if ext == ".md":
            labels.append("Clean_Markdown")
        elif ext == ".pdf":
            info = pdf_map.get(f, {})
            if info.get("has_error"):
                labels.append("Parse_Failed")
            elif info.get("type") in ("scan", "mixed"):
                labels.append("Scan_PDF")
            else:
                labels.append("Clean_Markdown")
        else:
            labels.append("Clean_Markdown")
        if not labels:
            labels.append("Parse_Failed")
        primary = labels[0]
        counter[primary] += 1
        docs.append({"path": f, "ext": ext, "labels": labels, "primary_label": primary})
    return {"summary": {"total": len(docs), "label_distribution": dict(counter)},
            "documents": docs}


def render_html(report, conflicts, sensitive, docs):
    def tbl(headers, rows):
        th = "".join(f"<th>{h}</th>" for h in headers)
        tb = "".join("<tr>" + "".join(f"<td>{str(c)[:100]}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>"
    fmt_rows = [[d["ext"], d["count"], f"{d['ratio']*100:.2f}%"]
                for d in report["format_stats"]["distribution"]]
    pdf_sum = report["pdf_classification"]
    pdf_rows = [["文字型", pdf_sum["text_type"]], ["扫描型", pdf_sum["scan_type"]],
                ["混合型", pdf_sum["mixed_type"]], ["解析失败", pdf_sum["parse_failed"]]]
    pct_rows = [[k, v] for k, v in report["length_stats"].get("percentiles", {}).items()]
    bin_rows = [[b["range"], b["count"], f"{b['ratio']*100:.2f}%"]
                for b in report["length_stats"].get("bins", [])]
    conflict_rows = [[c["file_a"], c["file_b"], c["hamming_distance"]] for c in conflicts[:100]]
    sens_rows = [[s["file"], s["type"], s["match"], s["context"][:60]] for s in sensitive[:100]]
    doc_rows = [[d["path"], d["primary_label"]] for d in docs[:200]]
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>文档质量评估报告</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;margin:30px;background:#f5f6f8}}
.card{{background:#fff;padding:20px;margin:20px 0;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
h1{{color:#1a1a1a}} h2{{color:#333;border-bottom:2px solid #4c8bf5;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #e0e0e0;padding:6px 10px;font-size:13px;text-align:left}}
th{{background:#f0f4fb}} .meta{{color:#666;font-size:13px}}</style></head><body>
<h1>文档质量评估报告</h1>
<p class="meta">文件夹：{report['meta']['folder_path']}<br>
总文件数：{report['meta']['total_files']}<br>耗时：{report['meta']['elapsed_seconds']} 秒</p>
<div class="card"><h2>一、格式分布</h2>{tbl(["扩展名","数量","占比"], fmt_rows)}</div>
<div class="card"><h2>二、PDF 类型识别</h2>{tbl(["类型","数量"], pdf_rows)}
<p class="meta">扫描型占比：{pdf_sum['scan_ratio_overall']*100:.2f}%</p></div>
<div class="card"><h2>三、文档长度分布</h2>{tbl(["分位数","字符数"], pct_rows)}<br>{tbl(["区间","数量","占比"], bin_rows)}</div>
<div class="card"><h2>四、重复检测</h2>
<p>MD5 精确重复组：{report['dedup']['exact_duplicate_groups']}，待确认相似对：{report['dedup']['pending_conflict_pairs']}</p>
{tbl(["文件A","文件B","汉明距离"], conflict_rows)}</div>
<div class="card"><h2>五、敏感信息检测</h2>
<p>总命中：{report['sensitive']['total_hits']}</p>{tbl(["文件","类型","匹配","上下文"], sens_rows)}</div>
<div class="card"><h2>六、分类标签</h2>
<p>{json.dumps(report['classification']['label_distribution'], ensure_ascii=False)}</p>
{tbl(["文件","主标签"], doc_rows)}</div>
</body></html>"""


def assess_folder(folder_path, config_path, output_dir):
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    cfg = load_config(config_path)
    print(f"[1/6] 配置加载：{config_path}")
    print(f"[2/6] 扫描文件夹：{folder_path}")
    supported = {".pdf", ".docx", ".md", ".txt", ".html", ".pptx", ".xlsx"}
    files = []
    for root, _, names in os.walk(folder_path):
        for n in names:
            if os.path.splitext(n)[1].lower() in supported:
                files.append(os.path.join(root, n))
    print(f"       发现 {len(files)} 个文件")
    print("[3/6] 格式分布统计...")
    fs = format_stats(files)
    print("[4/6] PDF 类型识别...")
    pdf_info = classify_pdfs(files, cfg["pdf_classifier"])
    print("[5/6] 长度分布 / 重复检测 / 敏感信息...")
    ls = length_stats(files, cfg["length_stats"])
    dd = dedup(files, cfg["dedup"])
    ss = scan_sensitive(files, cfg["sensitive"])
    print("[6/6] 分类标签 + 生成报告...")
    cls = classify_docs(files, pdf_info, dd, cfg.get("classifier", {}))
    report = {"meta": {"folder_path": folder_path, "total_files": len(files),
                       "elapsed_seconds": round(time.time()-t0, 2),
                       "config_path": config_path},
              "format_stats": fs, "pdf_classification": pdf_info["summary"],
              "length_stats": ls, "dedup": dd["summary"],
              "sensitive": ss["summary"], "classification": cls["summary"]}
    with open(os.path.join(output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "pending_conflicts.json"), "w", encoding="utf-8") as f:
        json.dump(dd["pending_conflicts"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "pending_sensitive.json"), "w", encoding="utf-8") as f:
        json.dump(ss["pending_items"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "classified_docs.json"), "w", encoding="utf-8") as f:
        json.dump(cls["documents"], f, ensure_ascii=False, indent=2)
    html = render_html(report, dd["pending_conflicts"], ss["pending_items"], cls["documents"])
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n完成！耗时 {report['meta']['elapsed_seconds']}s")
    print(f"报告目录：{output_dir}")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assess.py <folder_path> [output_dir] [config_path]")
        sys.exit(1)
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dqa_output"
    cfgp = sys.argv[3] if len(sys.argv) > 3 else "/ragflow/document_quality_assessment/assessment_config.yaml"
    assess_folder(folder, cfgp, out)