# -*- coding: utf-8 -*-
"""
DocumentQualityAssessmentSkill 核心引擎。
五大功能：格式分布统计 / PDF 文字型-扫描型-混合型分流 / 文档长度分布 /
        MD5+SimHash 重复检测 / 敏感信息检测；外加文档分类标签。
所有阈值来自 assessment_config.yaml，不在代码中硬编码。
"""
import os, re, json, hashlib, statistics
from collections import defaultdict, Counter

try:
    import yaml
except ImportError:
    yaml = None

from .simhash import SimHash

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assessment_config.yaml")


def load_config(path=CONFIG_PATH):
    if yaml is None or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- 文本提取 ----------------
def _read_text(path, ext):
    """尽力提取文档文本；失败返回空串与错误标记。"""
    if ext in (".md", ".txt"):
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read(), None
        except Exception as e:
            return "", str(e)
    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            texts = []
            for p in reader.pages:
                try:
                    texts.append(p.extract_text() or "")
                except Exception:
                    texts.append("")
            return "\n".join(texts), None
        except Exception as e:
            return "", f"PDF损坏/不可读: {e}"
    # docx/pptx/xlsx 等不深度解析，返回空（按扩展名统计即可）
    return "", None


def _pdf_page_chars(path):
    """返回 PDF 每页字符数列表；损坏返回 None。"""
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        out = []
        for p in reader.pages:
            try:
                out.append(len((p.extract_text() or "").strip()))
            except Exception:
                out.append(0)
        return out
    except Exception:
        return None


# ---------------- 功能1：格式分布统计 ----------------
def format_distribution(files, cfg):
    fmts = [f.lower() for f in cfg.get("formats", [])]
    counter = Counter()
    unknown = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in fmts:
            counter[ext] += 1
        else:
            counter[".(其它)"] += 1
            unknown.append(f)
    total = sum(counter.values())
    dist = {k: {"count": v, "ratio": round(v / total * 100, 2)} for k, v in counter.most_common()}
    return {"total": total, "distribution": dist, "unknown_examples": unknown[:10]}


# ---------------- 功能2：PDF 分类（文字/扫描/混合） ----------------
def classify_pdfs(files, cfg):
    pc = cfg.get("pdf_classification", {})
    char_th = pc.get("scan_page_char_threshold", 50)
    ratio_th = pc.get("scan_file_ratio_threshold", 0.7)
    results = []
    to_confirm = []
    for f in files:
        if not f.lower().endswith(".pdf"):
            continue
        pages = _pdf_page_chars(f)
        if pages is None:
            results.append({"file": f, "label": "Corrupt_PDF", "pages": 0, "scan_ratio": 0,
                            "note": "PDF 损坏，无法解析"})
            to_confirm.append({"file": f, "reason": "PDF 损坏或加密，需人工确认"})
            continue
        n = len(pages) or 1
        scan_pages = sum(1 for c in pages if c < char_th)
        scan_ratio = scan_pages / n
        if scan_ratio >= ratio_th:
            label = "Scan_PDF"
        elif scan_ratio <= 1 - ratio_th:
            label = "Text_PDF"
        else:
            label = "Mixed_PDF"
            to_confirm.append({"file": f, "reason": f"混合型（扫描页占比 {scan_ratio:.0%}），需人工确认处理方式",
                               "scan_ratio": round(scan_ratio, 3)})
        results.append({"file": f, "label": label, "pages": n, "scan_ratio": round(scan_ratio, 3),
                        "avg_page_chars": int(statistics.mean(pages)) if pages else 0})
    return {"pdfs": results, "to_confirm": to_confirm}


# ---------------- 功能3：文档长度分布 ----------------
def length_distribution(files, texts, cfg):
    la = cfg.get("length_analysis", {})
    buckets = la.get("buckets", [])
    pcts = la.get("percentiles", [25, 50, 75, 90, 99])
    lengths = {f: len(t) for f, t in texts.items()}
    vals = sorted(lengths.values())
    # 分位数
    def pct(p):
        if not vals:
            return 0
        k = (len(vals) - 1) * p / 100
        f, c = int(k), min(int(k) + 1, len(vals) - 1)
        return int(vals[f] + (vals[c] - vals[f]) * (k - f))
    percentiles = {f"P{p}": pct(p) for p in pcts}
    # 区间分布
    bucket_counts = {b["name"]: 0 for b in buckets}
    bucket_files = defaultdict(list)
    for f, v in lengths.items():
        for b in buckets:
            if b["min"] <= v < b["max"]:
                bucket_counts[b["name"]] += 1
                bucket_files[b["name"]].append(f)
                break
    return {
        "percentiles": percentiles,
        "mean": int(statistics.mean(vals)) if vals else 0,
        "buckets": bucket_counts,
        "outliers_long": [f for f in sorted(lengths, key=lengths.get, reverse=True)[:5]],
        "outliers_short": [f for f in sorted(lengths, key=lengths.get)[:5]],
    }


# ---------------- 功能4：重复检测（MD5 精确 + SimHash 近似） ----------------
def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def duplicate_detection(files, texts, cfg):
    dd = cfg.get("duplicate_detection", {})
    exact_groups = []
    if dd.get("enable_md5", True):
        by_md5 = defaultdict(list)
        for f in files:
            try:
                by_md5[file_md5(f)].append(f)
            except Exception:
                pass
        exact_groups = [g for g in by_md5.values() if len(g) > 1]

    version_conflicts = []  # 待确认
    if dd.get("enable_simhash", True):
        sh = SimHash(dd.get("simhash_hash_bits", 64))
        thr = dd.get("simhash_threshold", 3)
        fps = {}
        for f, t in texts.items():
            if len(t.strip()) >= 50:
                fps[f] = sh.hash(t)
        keys = list(fps.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = sh.hamming(fps[keys[i]], fps[keys[j]])
                if d <= thr:
                    # 截取相似片段上下文
                    version_conflicts.append({
                        "file_a": keys[i], "file_b": keys[j],
                        "hamming_distance": d,
                        "snippet_a": texts[keys[i]][:80],
                        "snippet_b": texts[keys[j]][:80],
                        "status": "待确认",
                    })
    return {"exact_duplicates": exact_groups, "version_conflicts_to_confirm": version_conflicts}


# ---------------- 功能5：敏感信息检测 ----------------
SENS_PATTERNS = {
    "phone": re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "id_card": re.compile(r"(?<!\d)(\d{17}[\dXx]|\d{15})(?!\d)"),
    "bank_card": re.compile(r"(?<!\d)(\d{16,19})(?!\d)"),
}


def sensitive_detection(files, texts, cfg):
    si = cfg.get("sensitive_info", {})
    ctx = si.get("context_chars", 20)
    enabled = {k: si.get(f"enable_{k}", False) for k in SENS_PATTERNS}
    findings = []  # 待审核列表，每条附上下文
    summary = Counter()
    for f, t in texts.items():
        for name, pat in SENS_PATTERNS.items():
            if not enabled.get(name):
                continue
            for m in pat.finditer(t):
                start = max(0, m.start() - ctx)
                end = min(len(t), m.end() + ctx)
                findings.append({
                    "file": f, "type": name, "value": m.group(),
                    "context": "..." + t[start:end].replace("\n", " ") + "...",
                    "status": "待审核",
                })
                summary[name] += 1
    return {"summary": dict(summary), "to_review": findings}


# ---------------- 分类标签 ----------------
def classify(files, pdf_info, cfg):
    labels = cfg.get("classification", {}).get("labels", [])
    route = cfg.get("classification", {}).get("route_map", {})
    pdf_label = {p["file"]: p["label"] for p in pdf_info.get("pdfs", [])}
    out = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if f in pdf_label:
            lab = pdf_label[f]
        elif ext in (".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"):
            lab = "Office_Doc"
        elif ext == ".md":
            lab = "Markdown"
        elif ext == ".txt":
            lab = "PlainText"
        elif ext in (".png", ".jpg", ".jpeg"):
            lab = "Image"
        elif ext == ".pdf":
            lab = "Text_PDF"
        else:
            lab = "Unknown"
        out[f] = {"label": lab, "route_to": route.get(lab, "fallback_parser")}
    return out


# ---------------- 主入口 ----------------
def assess(input_path, cfg=None, progress_cb=None):
    """对文件夹或文件列表执行质检，返回结构化报告 dict。"""
    cfg = cfg or load_config()
    if os.path.isdir(input_path):
        files = []
        for root, _, names in os.walk(input_path):
            for nm in names:
                files.append(os.path.join(root, nm))
    else:
        files = [input_path]

    # 提取文本（带错误处理）
    texts, errors = {}, []
    for i, f in enumerate(files):
        ext = os.path.splitext(f)[1].lower()
        t, err = _read_text(f, ext)
        texts[f] = t
        if err:
            errors.append({"file": f, "error": err})
        if progress_cb and i % 20 == 0:
            progress_cb(i, len(files))

    pdf_info = classify_pdfs(files, cfg)
    report = {
        "input": input_path,
        "file_count": len(files),
        "format_distribution": format_distribution(files, cfg),
        "pdf_classification": pdf_info,
        "length_distribution": length_distribution(files, texts, cfg),
        "duplicates": duplicate_detection(files, texts, cfg),
        "sensitive_info": sensitive_detection(files, texts, cfg),
        "classification": classify(files, pdf_info, cfg),
        "parse_errors": errors,
    }
    # 汇总“待确认/待审核”
    report["action_items"] = {
        "to_confirm_pdf": pdf_info["to_confirm"],
        "version_conflicts": report["duplicates"]["version_conflicts_to_confirm"],
        "sensitive_to_review": report["sensitive_info"]["to_review"],
    }
    return report


if __name__ == "__main__":
    import sys
    r = assess(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps(r, ensure_ascii=False, indent=2)[:2000])
