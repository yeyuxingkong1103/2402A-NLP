# -*- coding: utf-8 -*-
"""工单编号：人工智能NLP-RAG-实现文档质量评估Skill 并集成至智能体工作流工单

模块：document_quality_assessment.detectors
职责：五大功能的纯算法实现（不碰 IO、不碰并行），便于单元测试逐项验证。

  1. analyze_formats       格式分布统计
  2. classify_pdf          文字型 / 扫描型 / 混合型分流（按页）
  3. analyze_lengths       文档长度分布（P25/P50/P75/P90/P99 + 直方图）
  4. detect_duplicates     MD5 精确 + SimHash 近似（含待确认版本冲突列表）
  5. detect_sensitive      敏感信息检测（手机号 / 邮箱 / 身份证；银行卡默认关闭）

所有阈值均来自 Config（见 assessment_config.yaml），本模块不含阈值字面量。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .extractors import normalize_text

# ===========================================================================
# 1. 格式分布统计
# ===========================================================================


def analyze_formats(records: Sequence[dict[str, Any]], cfg: Any) -> dict[str, Any]:
    """统计扩展名与格式大类的数量、占比、体积。

    ``records``：逐文件记录，至少含 ``suffix`` / ``size_bytes``。
    """
    digits = cfg.integer("format_stats.ratio_digits", 4)
    group_map: dict[str, str] = {k.lower(): v for k, v in (cfg.get("format_stats.extension_groups", {}) or {}).items()}
    unknown_group = str(cfg.get("format_stats.unknown_group", "Other"))

    by_ext: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    bytes_by_group: Counter[str] = Counter()
    bytes_by_ext: Counter[str] = Counter()
    for rec in records:
        ext = str(rec.get("suffix", "") or "").lower() or "(无扩展名)"
        group = group_map.get(ext, unknown_group)
        size = int(rec.get("size_bytes", 0) or 0)
        by_ext[ext] += 1
        by_group[group] += 1
        bytes_by_ext[ext] += size
        bytes_by_group[group] += size

    total = len(records)
    total_bytes = sum(bytes_by_ext.values()) or 1

    def _rows(counter: Counter[str], byte_counter: Counter[str]) -> list[dict[str, Any]]:
        rows = []
        for name, count in counter.most_common():
            rows.append(
                {
                    "name": name,
                    "count": count,
                    "ratio": round(count / total, digits) if total else 0.0,
                    "bytes": byte_counter[name],
                    "bytes_ratio": round(byte_counter[name] / total_bytes, digits),
                }
            )
        return rows

    return {
        "total_files": total,
        "total_bytes": sum(bytes_by_ext.values()),
        "total_mb": round(sum(bytes_by_ext.values()) / 1048576, 2),
        "by_extension": _rows(by_ext, bytes_by_ext),
        "by_group": _rows(by_group, bytes_by_group),
    }


# ===========================================================================
# 2. PDF 文字型 / 扫描型 / 混合型分流
# ===========================================================================
PDF_TEXT = "Text_PDF"
PDF_SCAN = "Scan_PDF"
PDF_MIXED = "Mixed_PDF"
PDF_CORRUPTED = "Corrupted"


def classify_pdf(pages: Sequence[Any], page_count: int, cfg: Any) -> dict[str, Any]:
    """按页判定 PDF 类型。

    规则（全部阈值可配）：
      * 扫描页：该页有效字符数 < ``scan_detection.page_char_threshold``
      * 扫描页占比 > ``scan_ratio_threshold``            → Scan_PDF
      * 扫描页占比 >= ``mixed_ratio_threshold``           → Mixed_PDF
      * 否则                                              → Text_PDF
      * 抽检页数 < ``min_pages_for_confident_sample`` 且结论为 Scan/Mixed/Text 时，
        一定进"待确认"列表（启发式不保证严格正确）。

    ``pages`` 元素需含 ``char_count`` / ``is_scan`` 属性或同名键。
    """
    page_char_threshold = cfg.number("scan_detection.page_char_threshold")
    scan_ratio_threshold = cfg.number("scan_detection.scan_ratio_threshold")
    mixed_ratio_threshold = cfg.number("scan_detection.mixed_ratio_threshold")
    min_confident = cfg.integer("scan_detection.min_pages_for_confident_sample", 3)
    digits = cfg.integer("format_stats.ratio_digits", 4)

    def _get(pg: Any, key: str) -> Any:
        return pg.get(key) if isinstance(pg, dict) else getattr(pg, key)

    sampled = list(pages)
    n = len(sampled)
    if n == 0:
        return {
            "pdf_type": PDF_CORRUPTED,
            "scan_pages": 0,
            "low_text_pages": 0,
            "sampled_pages": 0,
            "page_count": int(page_count or 0),
            "scan_ratio": 0.0,
            "scan_ratio_literal": 0.0,
            "avg_chars_per_page": 0.0,
            "needs_confirm": True,
            "confirm_reason": "未抽取到任何页（0 页或全部页解析失败）",
        }

    scan_pages = 0
    low_text_pages = 0
    total_chars = 0
    for pg in sampled:
        chars = int(_get(pg, "char_count") or 0)
        total_chars += chars
        if chars < page_char_threshold:
            low_text_pages += 1
        flag = _get(pg, "is_scan")
        if flag is None:  # 兼容：调用方只给了 char_count
            flag = chars < page_char_threshold
        if flag:
            scan_pages += 1

    scan_ratio = scan_pages / n
    scan_ratio_literal = low_text_pages / n
    if scan_ratio > scan_ratio_threshold:
        pdf_type = PDF_SCAN
    elif scan_ratio >= mixed_ratio_threshold:
        pdf_type = PDF_MIXED
    else:
        pdf_type = PDF_TEXT

    needs_confirm = False
    reason = ""
    if n < min_confident:
        needs_confirm = True
        reason = f"仅抽检 {n} 页（< {min_confident} 页），样本不足以定论"
    elif scan_pages > 0 and pdf_type == PDF_SCAN and scan_ratio <= scan_ratio_threshold + 0.1:
        needs_confirm = True
        reason = f"扫描页占比 {scan_ratio:.4f} 紧贴阈值 {scan_ratio_threshold}，建议人工确认"
    elif pdf_type == PDF_MIXED:
        needs_confirm = True
        reason = (
            f"混合型（扫描页占比 {scan_ratio:.4f}，低文本页占比 {scan_ratio_literal:.4f}），"
            f"需确认哪些页缺文本层"
        )
    elif low_text_pages > scan_pages:
        needs_confirm = True
        reason = (
            f"存在 {low_text_pages - scan_pages} 个'低文本但非大图铺满'的页（附图/稀疏页），"
            f"建议确认是否需 OCR 回填"
        )

    return {
        "pdf_type": pdf_type,
        "scan_pages": scan_pages,
        "low_text_pages": low_text_pages,
        "sampled_pages": n,
        "page_count": int(page_count or n),
        "scan_ratio": round(scan_ratio, digits),
        "scan_ratio_literal": round(scan_ratio_literal, digits),
        "avg_chars_per_page": round(total_chars / n, 2),
        "needs_confirm": needs_confirm,
        "confirm_reason": reason,
    }


# ===========================================================================
# 3. 文档长度分布
# ===========================================================================


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数（numpy.percentile 默认口径），不依赖 numpy。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[int(pos)])
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def analyze_lengths(records: Sequence[dict[str, Any]], cfg: Any) -> dict[str, Any]:
    """字符数分位数 + 区间直方图 + 过短/过长计数。"""
    quantiles: list[float] = [float(q) for q in (cfg.get("length_stats.quantiles", []) or [])]
    names: list[str] = [str(s) for s in (cfg.get("length_stats.quantile_names", []) or [])]
    bins: list[dict[str, Any]] = list(cfg.get("length_stats.histogram_bins", []) or [])
    too_short = cfg.number("length_stats.too_short_chars")
    too_long = cfg.number("length_stats.too_long_chars")

    is_pdf = lambda r: str(r.get("suffix", "")).lower() == ".pdf"  # noqa: E731

    def _collect(pred) -> list[int]:
        out = []
        for rec in records:
            if rec.get("corrupted"):
                continue
            if pred(rec):
                out.append(int(rec.get("char_count", 0) or 0))
        return sorted(out)

    all_chars = _collect(lambda r: True)
    pdf_chars = _collect(is_pdf)
    non_pdf_chars = _collect(lambda r: not is_pdf(r))

    def _describe(values: list[int]) -> dict[str, Any]:
        if not values:
            return {
                "count": 0, "sum": 0, "mean": 0.0, "min": 0, "max": 0,
                "quantiles": {n: 0 for n in names}, "histogram": [],
                "too_short": 0, "too_long": 0,
            }
        quant = {n: round(percentile(values, q), 2) for n, q in zip(names, quantiles)}
        hist = []
        for spec in bins:
            lo = spec.get("lo", 0)
            hi = spec.get("hi", None)
            if hi is None:
                count = sum(1 for v in values if v >= lo)
            else:
                count = sum(1 for v in values if lo <= v < hi)
            hist.append({"label": str(spec.get("label", f"{lo}+")), "lo": lo, "hi": hi, "count": count,
                         "ratio": round(count / len(values), 4)})
        return {
            "count": len(values),
            "sum": int(sum(values)),
            "mean": round(sum(values) / len(values), 2),
            "min": int(values[0]),
            "max": int(values[-1]),
            "quantiles": quant,
            "histogram": hist,
            "too_short": sum(1 for v in values if v < too_short),
            "too_long": sum(1 for v in values if v > too_long),
        }

    return {
        "scope": "all_parseable_files",
        "thresholds": {"too_short_chars": too_short, "too_long_chars": too_long, "quantiles": quantiles},
        "all": _describe(all_chars),
        "pdf_only": _describe(pdf_chars),
        "non_pdf": _describe(non_pdf_chars),
    }


# ===========================================================================
# 4. 重复检测：MD5 精确 + SimHash 近似
# ===========================================================================
_TOKEN_LATIN_RE = re.compile(r"[a-z0-9]+")
_TOKEN_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """分词：拉丁/数字按词、CJK 按字。

    与 jieba 相比在本语料（中文 PDF 混排）上更稳、且零依赖；
    对比见 优化/优化说明.md。
    """
    if not text:
        return []
    lower = text.lower()
    latin = _TOKEN_LATIN_RE.findall(lower)
    cjk = _TOKEN_CJK_RE.findall(lower)
    return latin + cjk


def _md5_of_file(path: Path, read_bytes: int = 0) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        if read_bytes and read_bytes > 0:
            h.update(fh.read(read_bytes))
        else:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def md5_of_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def detect_md5_duplicates(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 MD5 分组，输出重复组（首份为"保留"，其余为"重复"）。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        digest = rec.get("md5")
        if not digest or rec.get("corrupted"):
            continue
        groups[str(digest)].append(rec)
    out = []
    for digest, items in groups.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda r: str(r.get("path", "")))
        out.append(
            {
                "md5": digest,
                "count": len(items),
                "keep": items[0].get("path"),
                "files": [r.get("path") for r in items],
                "size_bytes": int(items[0].get("size_bytes", 0) or 0),
                "wasted_bytes": int(items[0].get("size_bytes", 0) or 0) * (len(items) - 1),
            }
        )
    out.sort(key=lambda g: (-g["count"], str(g["md5"])))
    return out


def simhash64(text: str, cfg: Any) -> int:
    """自实现 64 位 SimHash：分词 → 加权 → 64 维向量累加 → 降维取符号位。

    步骤（与经典 SimHash 一致）：
      1. 文本归一化并按配置截断；
      2. 切成 k-gram 分片（shingle），分片频次即权重；
      3. 每个分片取 md5 前 8 字节 = 64 位哈希；
      4. 逐位累加：位为 1 加权重、为 0 减权重；
      5. 每位 >0 记 1，否则记 0 → 64 位整数。
    """
    max_chars = cfg.integer("duplicate_detection.simhash.max_chars", 20000)
    shingle_size = cfg.integer("duplicate_detection.simhash.shingle_size", 4)
    shingle_step = max(1, cfg.integer("duplicate_detection.simhash.shingle_step", 2))
    bits = cfg.integer("duplicate_detection.simhash.bits", 64)

    norm = normalize_text(text)
    if max_chars > 0:
        norm = norm[:max_chars]
    tokens = tokenize(norm)
    if not tokens:
        return 0
    if len(tokens) < shingle_size:
        shingles = [" ".join(tokens)]
    else:
        shingles = [
            " ".join(tokens[i : i + shingle_size])
            for i in range(0, len(tokens) - shingle_size + 1, shingle_step)
        ]
    weights = Counter(shingles)

    vector = [0] * bits
    for shingle, weight in weights.items():
        h = int.from_bytes(hashlib.md5(shingle.encode("utf-8")).digest()[: bits // 8], "big")
        for b in range(bits):
            if (h >> b) & 1:
                vector[b] += weight
            else:
                vector[b] -= weight
    fingerprint = 0
    for b in range(bits):
        if vector[b] > 0:
            fingerprint |= 1 << b
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _candidate_pairs(
    docs: list[tuple[int, int, int]],
    *,
    prefilter: bool,
    tolerance: float,
    bits: int,
) -> set[tuple[int, int]]:
    """候选对生成：按长度预筛 + 16 位分带（4 band）索引。

    ``docs``：``[(doc_index, simhash, char_count), ...]``
    返回需要精确计算海明距离的下标对集合，避免 O(n²) 全比。
    """
    pairs: set[tuple[int, int]] = set()
    n = len(docs)
    if n < 2:
        return pairs

    band_bits = 16
    n_bands = max(1, bits // band_bits)
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(n_bands)]
    for pos, (_idx, fp, _chars) in enumerate(docs):
        for b in range(n_bands):
            key = (fp >> (b * band_bits)) & ((1 << band_bits) - 1)
            buckets[b][key].append(pos)

    for b in range(n_bands):
        for members in buckets[b].values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, c = members[i], members[j]
                    if prefilter:
                        ca, cc = docs[a][2], docs[c][2]
                        denom = max(ca, cc, 1)
                        if abs(ca - cc) / denom > tolerance:
                            continue
                    pairs.add((a, c) if a < c else (c, a))
    return pairs


def _similar_snippets(text_a: str, text_b: str, count: int, chars: int) -> list[dict[str, str]]:
    """相似片段对照：用 shingle 集合交集找出共同片段所在的原文位置，截取上下文。"""
    if count <= 0 or chars <= 0:
        return []
    ta, tb = tokenize(text_a), tokenize(text_b)
    if not ta or not tb:
        return []
    window = max(4, chars // 4)
    set_b = {" ".join(tb[i : i + window]) for i in range(0, max(0, len(tb) - window), 3)}
    hits: list[dict[str, str]] = []
    step = max(1, window // 2)
    for i in range(0, max(0, len(ta) - window), step):
        key = " ".join(ta[i : i + window])
        if key in set_b:
            hits.append({"a": key[:chars], "b": key[:chars]})
            if len(hits) >= count:
                break
    if not hits:
        # 退化：无精确共同分片时，给出两侧开头作为可读对照
        hits.append({"a": normalize_text(text_a)[:chars], "b": normalize_text(text_b)[:chars]})
    return hits


def detect_simhash_conflicts(
    records: Sequence[dict[str, Any]], cfg: Any, *, texts: dict[str, str] | None = None
) -> dict[str, Any]:
    """SimHash 近似重复检测 → "待确认的版本冲突列表"。"""
    enabled = cfg.boolean("duplicate_detection.simhash.enabled", True)
    if not enabled:
        return {"enabled": False, "candidates_compared": 0, "conflicts": [], "total_conflicts": 0}

    threshold = cfg.integer("duplicate_detection.simhash.hamming_threshold", 3)
    bits = cfg.integer("duplicate_detection.simhash.bits", 64)
    prefilter = cfg.boolean("duplicate_detection.prefilter_by_length", True)
    tolerance = cfg.number("duplicate_detection.length_ratio_tolerance", 0.25)
    snippet_count = cfg.integer("duplicate_detection.simhash.snippet_count", 3)
    snippet_chars = cfg.integer("duplicate_detection.simhash.snippet_chars", 80)
    max_items = cfg.integer("duplicate_detection.max_conflict_items", 0)
    digits = cfg.integer("format_stats.ratio_digits", 4)

    docs: list[tuple[int, int, int]] = []
    usable: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("corrupted"):
            continue
        fp = rec.get("simhash")
        if fp is None:
            continue
        chars = int(rec.get("char_count", 0) or 0)
        if chars <= 0:
            continue
        usable.append(rec)
        docs.append((len(docs), int(fp), chars))

    pairs = _candidate_pairs(docs, prefilter=prefilter, tolerance=tolerance, bits=bits)
    conflicts: list[dict[str, Any]] = []
    md5_pairs: set[tuple[str, str]] = set()
    for a, b in sorted(pairs):
        rec_a, rec_b = usable[a], usable[b]
        dist = hamming_distance(int(docs[a][1]), int(docs[b][1]))
        if rec_a.get("md5") and rec_a.get("md5") == rec_b.get("md5"):
            md5_pairs.add((str(rec_a.get("path")), str(rec_b.get("path"))))
        if dist > threshold:
            continue
        similarity = round(1 - dist / bits, digits)
        item = {
            "file_a": rec_a.get("path"),
            "file_b": rec_b.get("path"),
            "hamming_distance": dist,
            "similarity": similarity,
            "char_count_a": int(rec_a.get("char_count", 0) or 0),
            "char_count_b": int(rec_b.get("char_count", 0) or 0),
            "size_bytes_a": int(rec_a.get("size_bytes", 0) or 0),
            "size_bytes_b": int(rec_b.get("size_bytes", 0) or 0),
            "md5_identical": bool(rec_a.get("md5") and rec_a.get("md5") == rec_b.get("md5")),
            "reason": (
                f"海明距离 {dist} ≤ 阈值 {threshold}（SimHash 相似度 {similarity:.4f}）"
                + ("；且 MD5 完全相同" if rec_a.get("md5") and rec_a.get("md5") == rec_b.get("md5") else "")
            ),
            "need_manual_confirm": True,
            "similar_snippets": _similar_snippets(
                (texts or {}).get(str(rec_a.get("path")), ""),
                (texts or {}).get(str(rec_b.get("path")), ""),
                snippet_count,
                snippet_chars,
            )
            if texts
            else [],
        }
        conflicts.append(item)

    conflicts.sort(key=lambda x: (x["hamming_distance"], str(x["file_a"]), str(x["file_b"])))
    total = len(conflicts)
    if max_items and max_items > 0:
        conflicts = conflicts[:max_items]

    return {
        "enabled": True,
        "hamming_threshold": threshold,
        "bits": bits,
        "shingle_size": cfg.integer("duplicate_detection.simhash.shingle_size", 4),
        "length_ratio_tolerance": tolerance,
        "candidates_compared": len(pairs),
        "comparisons_if_brute_force": len(docs) * (len(docs) - 1) // 2,
        "conflicts": conflicts,
        "total_conflicts": total,
        "truncated": bool(max_items and total > max_items),
    }


def detect_duplicates(
    records: Sequence[dict[str, Any]], cfg: Any, *, texts: dict[str, str] | None = None
) -> dict[str, Any]:
    """重复检测总入口：MD5 分层 + SimHash 分层。"""
    md5_groups = detect_md5_duplicates(records)
    max_groups = cfg.integer("duplicate_detection.max_md5_groups", 0)
    total_groups = len(md5_groups)
    if max_groups and max_groups > 0:
        md5_groups = md5_groups[:max_groups]
    simhash = detect_simhash_conflicts(records, cfg, texts=texts)
    return {
        "md5_exact": {
            "enabled": True,
            "duplicate_groups": md5_groups,
            "total_groups": total_groups,
            "duplicate_file_count": sum(g["count"] - 1 for g in md5_groups),
            "wasted_bytes": sum(g["wasted_bytes"] for g in md5_groups),
        },
        "simhash_near": simhash,
        # 工单要求的"待确认的版本冲突列表"= SimHash 高相似对（已按海明距离排序）
        "pending_version_conflicts": simhash["conflicts"],
        "pending_version_conflicts_total": simhash["total_conflicts"],
    }


# ===========================================================================
# 5. 敏感信息检测
# ===========================================================================

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK_CHARS = "10X98765432"


def id_card_checksum_ok(value: str) -> bool:
    """GB 11643-1999 18 位身份证校验码校验。"""
    if len(value) != 18:
        return False
    body, check = value[:17], value[17].upper()
    if not body.isdigit():
        return False
    total = sum(int(body[i]) * _ID_WEIGHTS[i] for i in range(17))
    return _ID_CHECK_CHARS[total % 11] == check


def luhn_ok(value: str) -> bool:
    """Luhn 校验（银行卡）。"""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_repeat_digits(value: str) -> bool:
    return len(set(value)) <= 2


@dataclass
class SensitiveHit:
    rule: str
    type: str
    value: str
    severity: str
    contexts: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "type": self.type,
            "value": self.value,
            "severity": self.severity,
            "contexts": self.contexts,
            "start": self.start,
            "end": self.end,
        }


def detect_sensitive_in_text(text: str, cfg: Any) -> list[dict[str, Any]]:
    """对单段文本跑全部已启用规则，返回待审核条目（含上下文片段）。"""
    if not cfg.boolean("sensitive_detection.enabled", True) or not text:
        return []
    context_chars = cfg.integer("sensitive_detection.context_chars", 30)
    max_contexts = cfg.integer("sensitive_detection.max_contexts_per_hit", 2)
    scan_max_chars = cfg.integer("sensitive_detection.scan_max_chars", 0)
    max_hits = cfg.integer("sensitive_detection.max_hits_per_file_per_rule", 0)

    body = text if scan_max_chars <= 0 else text[:scan_max_chars]
    rules = cfg.get("sensitive_detection.rules", []) or []
    hits: list[dict[str, Any]] = []
    for rule in rules:
        if not rule.get("enabled", False):
            continue
        pattern = rule.get("pattern")
        if not pattern:
            continue
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        count = 0
        for m in regex.finditer(body):
            if max_hits and count >= max_hits:
                break
            value = m.group(0)
            # 二次校验，显著降低误报
            if rule.get("validate_repeat_digits") and _is_repeat_digits(value):
                continue
            if rule.get("validate_checksum") and rule.get("type") == "id_card_cn" and len(value) == 18:
                if not id_card_checksum_ok(value):
                    continue
            if rule.get("validate_luhn") and not luhn_ok(value):
                continue
            start, end = m.start(), m.end()
            ctx_lo = max(0, start - context_chars)
            ctx_hi = min(len(body), end + context_chars)
            snippet = body[ctx_lo:ctx_hi].replace("\n", " ")
            marker = f"{body[ctx_lo:start]}【{value}】{body[end:ctx_hi]}".replace("\n", " ")
            hit = SensitiveHit(
                rule=str(rule.get("name", rule.get("type", "未知规则"))),
                type=str(rule.get("type", "")),
                value=value,
                severity=str(rule.get("severity", "medium")),
                contexts=[marker] if max_contexts > 0 else [],
                start=start,
                end=end,
            )
            d = hit.to_dict()
            d["context_raw"] = snippet
            hits.append(d)
            count += 1
    return hits


def detect_sensitive(
    records: Sequence[dict[str, Any]], cfg: Any, *, texts: dict[str, str] | None = None
) -> dict[str, Any]:
    """敏感信息总入口 → "待审核列表"。"""
    texts = texts or {}
    review: list[dict[str, Any]] = []
    by_rule: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    files_with_hits = 0
    enabled_rules = [str(r.get("name")) for r in (cfg.get("sensitive_detection.rules", []) or []) if r.get("enabled")]
    disabled_rules = [str(r.get("name")) for r in (cfg.get("sensitive_detection.rules", []) or []) if not r.get("enabled")]
    scan_filenames = cfg.boolean("sensitive_detection.scan_filenames", False)

    for rec in records:
        path = str(rec.get("path", ""))
        payload = texts.get(path, "")
        if scan_filenames and path:
            payload = f"{Path(path).name} {payload}"
        hits = detect_sensitive_in_text(payload, cfg)
        if not hits:
            continue
        files_with_hits += 1
        for h in hits:
            by_rule[h["rule"]] += 1
            by_type[h["type"]] += 1
        review.append(
            {
                "file": path,
                "hit_count": len(hits),
                "hits": hits,
                "need_manual_review": True,
            }
        )

    max_items = cfg.integer("sensitive_detection.max_review_items", 0)
    total_files = len(review)
    if max_items and max_items > 0:
        review = review[:max_items]

    return {
        "enabled": cfg.boolean("sensitive_detection.enabled", True),
        "enabled_rules": enabled_rules,
        "disabled_rules": disabled_rules,
        "by_rule": dict(by_rule),
        "by_type": dict(by_type),
        "files_with_hits": files_with_hits,
        "total_hits": sum(by_rule.values()),
        "pending_review": review,
        "pending_review_total": total_files,
        "truncated": bool(max_items and total_files > max_items),
    }


# ===========================================================================
# 6. 分类标签体系
# ===========================================================================
def assign_tags(rec: dict[str, Any], cfg: Any) -> tuple[list[str], str]:
    """按配置的 priority 顺序给单个文件打全部命中标签（顺序即优先级）。

    返回 ``(tags, primary_tag)``：``tags`` 为全部命中标签（按 priority 排序），
    ``primary_tag`` 为决定路由分支的主标签。
    """
    priority: list[str] = [str(t) for t in (cfg.get("tagging.priority", []) or [])]
    defs: dict[str, Any] = cfg.get("tagging.definitions", {}) or {}
    too_short = cfg.number("length_stats.too_short_chars")
    too_long = cfg.number("length_stats.too_long_chars")

    suffix = str(rec.get("suffix", "")).lower()
    group_map = {k.lower(): v for k, v in (cfg.get("format_stats.extension_groups", {}) or {}).items()}
    group = group_map.get(suffix, str(cfg.get("format_stats.unknown_group", "Other")))
    chars = int(rec.get("char_count", 0) or 0)
    size = int(rec.get("size_bytes", 0) or 0)
    pdf_type = rec.get("pdf_type")

    matched: list[str] = []
    for tag in priority:
        spec = defs.get(tag) or {}
        kind = spec.get("kind")
        hit = False
        if kind == "corrupted":
            hit = bool(rec.get("corrupted"))
        elif kind == "md5_duplicate":
            hit = bool(rec.get("is_md5_duplicate"))
        elif kind == "has_sensitive":
            hit = int(rec.get("sensitive_hits", 0) or 0) > 0
        elif kind == "pdf_scan":
            hit = pdf_type == PDF_SCAN
        elif kind == "pdf_mixed":
            hit = pdf_type == PDF_MIXED
        elif kind == "pdf_text":
            hit = pdf_type == PDF_TEXT
        elif kind == "format_group":
            hit = group == spec.get("match_group")
        elif kind == "too_short":
            hit = (not rec.get("corrupted")) and chars < too_short and not rec.get("pure_scan")
        elif kind == "too_long":
            hit = (not rec.get("corrupted")) and chars > too_long
        elif kind == "empty":
            hit = (not rec.get("corrupted")) and chars == 0 and size > 0 and not rec.get("pure_scan")
        elif kind == "normal":
            hit = not matched
        if hit:
            matched.append(tag)
    if not matched:
        matched = ["Normal"]
    # 去重保序
    seen: set[str] = set()
    ordered = [t for t in matched if not (t in seen or seen.add(t))]
    primary = ordered[0]
    if primary in ("Too_Short", "Too_Long", "Empty") and len(ordered) > 1:
        # 主标签优先取"格式/类型"类标签，异常标签保留在 tags 中
        for t in ordered:
            if t not in ("Too_Short", "Too_Long", "Empty", "Normal"):
                primary = t
                break
    return ordered, primary


def tag_distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """标签分布统计（主标签 + 全标签）。"""
    primary = Counter(str(r.get("primary_tag", "Normal")) for r in records)
    all_tags: Counter[str] = Counter()
    for rec in records:
        for t in rec.get("tags", []) or []:
            all_tags[str(t)] += 1
    total = len(records) or 1
    return {
        "by_primary": [
            {"name": k, "count": v, "ratio": round(v / total, 4)} for k, v in primary.most_common()
        ],
        "by_all_tags": [
            {"name": k, "count": v, "ratio": round(v / total, 4)} for k, v in all_tags.most_common()
        ],
    }
