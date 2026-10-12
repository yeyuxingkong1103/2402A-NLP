# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-实现文档质量评估Skill并集成至智能体工作流工单
文档质量评估核心逻辑：五大功能 + 分类标签 + 结构化报告。
"""
import hashlib
import os
import re
import statistics

try:
    import yaml
except Exception:
    yaml = None

try:
    import pymupdf
except Exception:
    pymupdf = None


def _load_config(cfg_path=None):
    cfg_path = cfg_path or os.path.join(os.path.dirname(__file__), "assessment_config.yaml")
    if yaml and os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    # 默认配置（yaml 不可用时）
    return {
        "scan_page_char_threshold": 100,
        "scanned_ratio_threshold": 0.7,
        "length_percentiles": [25, 50, 75, 90, 99],
        "length_bins": [0, 1000, 5000, 10000, 50000, 100000, 1000000],
        "simhash_hamming_threshold": 3,
        "simhash_bits": 64,
        "detect_phone": True, "detect_email": True,
        "detect_id_card": True, "detect_bankcard": False,
        "context_window": 40,
    }


# ---- 敏感信息正则 ----
PHONE_RE = re.compile(r"1[3-9]\d{9}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
ID_CARD_RE = re.compile(r"\d{17}[\dXx]|\d{15}")
BANKCARD_RE = re.compile(r"\d{16,19}")


def _context(text, m, window):
    s = max(m.start() - window, 0)
    e = min(m.end() + window, len(text))
    return text[s:e]


def _tokens(text):
    return re.findall(r"[一-龥A-Za-z0-9]{2,4}", text)


def simhash(text, bits=64):
    v = [0] * bits
    for tok in set(_tokens(text)):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum((1 << i) for i in range(bits) if v[i] > 0)


def hamming(a, b):
    return bin(a ^ b).count("1")


class DocumentQualityAssessor:
    def __init__(self, cfg_path=None):
        self.cfg = _load_config(cfg_path)

    # 1. 格式分布统计
    def format_distribution(self, files):
        from collections import Counter
        ext = Counter(os.path.splitext(f)[1].lower() or "(无扩展名)" for f in files)
        total = len(files)
        return {e: {"count": c, "ratio": round(c / total, 4)} for e, c in ext.most_common()}

    # 2. PDF 页面类型识别
    def pdf_page_type(self, path):
        if not pymupdf:
            return {"type": "Unknown", "scanned_pages": [], "reason": "pymupdf 未安装"}
        try:
            doc = pymupdf.open(path)
        except Exception as e:
            return {"type": "Error", "scanned_pages": [], "reason": f"损坏文件: {e}"}
        scanned = []
        total = 0
        for pno, page in enumerate(doc, 1):
            total += 1
            if len(page.get_text().strip()) < self.cfg["scan_page_char_threshold"]:
                scanned.append(pno)
        doc.close()
        ratio = len(scanned) / max(total, 1)
        if ratio > self.cfg["scanned_ratio_threshold"]:
            ptype = "Scan_PDF"
        elif scanned:
            ptype = "Mixed_PDF"
        else:
            ptype = "Text_PDF"
        return {"type": ptype, "scanned_pages": scanned, "scanned_ratio": round(ratio, 4)}

    # 3. 文档长度分布
    def length_distribution(self, texts):
        lengths = sorted(len(t) for t in texts)
        n = len(lengths)
        if n == 0:
            return {"percentiles": {}, "bins": {}}
        pct = {f"P{p}": lengths[min(int(n * p / 100), n - 1)]
               for p in self.cfg["length_percentiles"]}
        bins = {}
        b = self.cfg["length_bins"]
        for i in range(len(b) - 1):
            cnt = sum(1 for L in lengths if b[i] <= L < b[i + 1])
            bins[f"{b[i]}-{b[i+1]}"] = cnt
        return {"percentiles": pct, "bins": bins}

    # 4. 重复检测（MD5 精确 + SimHash 相似）
    def duplicates(self, files, contents):
        md5_map = {}
        exact = []
        for f, c in zip(files, contents):
            h = hashlib.md5(c.encode("utf-8", errors="ignore")).hexdigest()
            md5_map.setdefault(h, []).append(f)
        for h, fs in md5_map.items():
            if len(fs) > 1:
                exact.append({"md5": h, "files": fs, "kind": "exact"})

        sim = []
        sig = [(f, simhash(c, self.cfg["simhash_bits"])) for f, c in zip(files, contents)]
        for i in range(len(sig)):
            for j in range(i + 1, len(sig)):
                if hamming(sig[i][1], sig[j][1]) <= self.cfg["simhash_hamming_threshold"]:
                    sim.append({"files": [sig[i][0], sig[j][0]],
                                "distance": hamming(sig[i][1], sig[j][1]),
                                "kind": "similar"})
        return {"exact": exact, "version_conflicts": sim}

    # 5. 敏感信息检测（附上下文）
    def sensitive_info(self, files, contents):
        hits = []
        for f, c in zip(files, contents):
            for name, pat, enabled in [
                ("phone", PHONE_RE, self.cfg["detect_phone"]),
                ("email", EMAIL_RE, self.cfg["detect_email"]),
                ("id_card", ID_CARD_RE, self.cfg["detect_id_card"]),
                ("bankcard", BANKCARD_RE, self.cfg["detect_bankcard"]),
            ]:
                if not enabled:
                    continue
                for m in pat.finditer(c):
                    hits.append({
                        "file": f, "type": name, "value": m.group(),
                        "context": _context(c, m, self.cfg["context_window"]),
                    })
        return hits

    # 总入口
    def assess(self, folder):
        files = []
        for root, _, names in os.walk(folder):
            for n in names:
                files.append(os.path.join(root, n))
        return self.assess_files(files)

    def assess_files(self, files):
        contents = []
        pdf_types = {}
        labels = []
        total = len(files)
        for idx, f in enumerate(files, 1):
            # 进度反馈（处理 1700 份文档耗时较长）
            if idx % 200 == 0 or idx == total:
                print(f"[assess] 进度 {idx}/{total}")
            try:
                if f.lower().endswith(".pdf"):
                    c = self._pdf_text(f)
                    pdf_types[os.path.basename(f)] = self.pdf_page_type(f)
                    labels.append({"file": f, "label": pdf_types[os.path.basename(f)]["type"]})
                else:
                    with open(f, encoding="utf-8", errors="ignore") as fh:
                        c = fh.read()
                    labels.append({"file": f, "label": "Other"})
                contents.append(c)
            except Exception as e:
                contents.append("")
                labels.append({"file": f, "label": "Error", "reason": str(e)})

        report = {
            "total_files": total,
            "format_distribution": self.format_distribution(files),
            "pdf_types": pdf_types,
            "length_distribution": self.length_distribution(contents),
            "duplicates": self.duplicates(files, contents),
            "sensitive_info": self.sensitive_info(files, contents),
            "labels": labels,
        }
        return report

    def _pdf_text(self, path):
        if not pymupdf:
            return ""
        try:
            doc = pymupdf.open(path)
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
            return text
        except Exception:
            return ""
