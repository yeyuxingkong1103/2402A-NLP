# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：任务四的验证与调优脚本。五项检查——① 分类准确性抽样交叉验证（用与字符数判定
#   独立的“图像覆盖率 + 字体资源表”信号）；② 敏感信息正控制样本；③ 扫描件 OCR 路由补位实测；
#   ④ 版本冲突候选著录项核查；⑤ 分类标签合理性审查。汇总输出 evaluation/验证报告.md。

import os                                                          # 路径
import sys                                                         # 参数
import json                                                        # 读写报告
import random                                                      # 分层抽样
import argparse                                                    # 命令行

import fitz                                                        # 独立复检

BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # 工单根目录
sys.path.insert(0, BASE_DIR)                                       # 便于直接运行

from document_quality_assessment import sensitive as sens_mod      # 复用格式校验与检测
from document_quality_assessment import load_config                # 配置
from document_ingestion_workflow import DocumentIngestionWorkflow  # 工作流（OCR 路由）
from document_ingestion_workflow import parsers                    # 解析器注册表
import verify_report                                               # 报告渲染

IMG_RATIO_HINT = 0.30                                              # “该页由图像主导”的阈值


def independent_signal(path):
    """与字符数判定相互独立的复检信号：大图页占比 + 无字体页占比。"""
    doc = fitz.open(path)                                          # 打开
    try:
        pages = doc.page_count                                     # 页数
        if pages == 0:
            return None
        img_pages, fontless = 0, 0                                 # 大图页 / 无字体页
        for i in range(pages):                                     # 逐页
            page = doc[i]
            area = max(page.rect.width * page.rect.height, 1.0)     # 页面面积
            biggest = 0.0                                          # 最大图像占比
            for img in page.get_images(full=True):                 # 遍历嵌入图像
                try:
                    bbox = page.get_image_bbox(img)                # 图像位置
                except Exception:
                    continue
                if not bbox.is_empty and not bbox.is_infinite:
                    biggest = max(biggest, abs(bbox.width * bbox.height) / area)
            if biggest >= IMG_RATIO_HINT:                          # 该页由大图主导
                img_pages += 1
            if not page.get_fonts():                               # 该页无任何字体资源
                fontless += 1
        return {"pages": pages,
                "big_image_page_ratio": round(img_pages / float(pages), 4),
                "fontless_page_ratio": round(fontless / float(pages), 4)}
    finally:
        doc.close()


def visual_like(sig):
    """独立信号给出的“是否图像型”结论。"""
    if not sig:
        return None
    return (sig["big_image_page_ratio"] >= 0.5) or (sig["fontless_page_ratio"] >= 0.5)


def stratified_sample(docs, per_label, seed=20260123):
    """按主标签分层抽样，保证每个标签都有代表样本；固定种子保证可复现。"""
    rnd = random.Random(seed)                                      # 固定随机种子
    buckets = {}                                                   # 标签 → 文档
    for d in docs:
        buckets.setdefault(d["primary_label"], []).append(d)
    picked = []                                                    # 抽样结果
    for label in sorted(buckets):                                  # 逐标签
        items = buckets[label][:]                                  # 复制后打乱
        rnd.shuffle(items)
        picked.extend(items[:per_label])                           # 取前 N 个
    return picked


def check_classification(report, per_label):
    """抽样交叉验证 PDF 分类，返回 (明细, 汇总)。"""
    pdfs = [d for d in report["classification"]["documents"]
            if (d.get("ext") or "") == ".pdf" and d.get("primary_label") != "Other"]
    rows, agree, total = [], 0, 0                                  # 明细与计数
    for d in stratified_sample(pdfs, per_label):                   # 逐样本独立复检
        try:
            sig = independent_signal(d["path"])                    # 独立信号
        except Exception as exc:                                   # 复检失败记为不一致，不掩盖问题
            rows.append({"name": d["name"], "label": d["primary_label"],
                         "signal": "复检失败：%s" % exc, "agree": False})
            total += 1
            continue
        label_text = d["primary_label"] in ("Text_PDF", "Mixed_PDF")   # 标签偏向文字型
        sig_text = not visual_like(sig)                            # 独立信号偏向文字型
        ok = (label_text == sig_text)                              # 两者是否一致
        agree += 1 if ok else 0
        total += 1
        rows.append({"name": d["name"], "label": d["primary_label"],
                     "scan_ratio": d.get("scan_ratio"),
                     "big_image_page_ratio": sig["big_image_page_ratio"],
                     "fontless_page_ratio": sig["fontless_page_ratio"],
                     "signal": "文字型" if sig_text else "图像型", "agree": ok})
    rate = round(agree / float(total), 4) if total else 0.0        # 一致率
    return rows, {"sample_size": total, "agree": agree, "rate": rate,
                  "threshold": IMG_RATIO_HINT}


def check_control(cfg):
    """正控制样本：把构造的敏感信息注入真实专利正文，验证能否全部检出。"""
    doc_dir = cfg.get("paths.default_target")                      # 数据集目录
    sample = None                                                  # 取一份真实文字型专利
    for name in sorted(os.listdir(doc_dir)):                       # 找一份文本层足够的文档
        if not name.lower().endswith(".pdf"):
            continue
        p = os.path.join(doc_dir, name)
        try:
            d = fitz.open(p)
            text = "".join(d[i].get_text() for i in range(min(d.page_count, 3)))
            d.close()
        except Exception:
            continue
        if len(text) > 1500:                                       # 文本层足够
            sample = (name, text)
            break
    if sample is None:                                             # 找不到就跳过
        return {"rows": [], "found": 0, "total": 0, "rate": 0.0, "source": ""}
    name, text = sample                                            # 真实正文
    injected = [("手机号", "13812345678"),                          # 构造的四类值
                ("邮箱", "zhangsan.patent@example.com"),
                ("身份证号", "11010519491231002X"),                 # 校验位正确
                ("银行卡号", "6222021234567894")]                   # 需配置开启才会检出（过 Luhn 校验）
    cfg2 = load_config()[0]                                        # 复制一份配置
    cfg2._data["sensitive"]["enabled_types"]["bank_card"] = True    # 本次验证开启银行卡
    marked = "联系信息如下：" + "，".join("%s%s" % (t, v) for t, v in injected) + "。"
    probe_text = text[:800] + marked + text[800:]                  # 注入到真实正文中
    hits = sens_mod.detect(probe_text, cfg2)                       # 检测
    rows, found = [], 0                                            # 明细与命中数
    for t, v in injected:                                          # 逐条核对
        hit = next((h for h in hits if h["type"] == t and h["value"] == v), None)
        found += 1 if hit else 0
        rows.append({"type": t, "value": v, "found": bool(hit),
                     "context": hit["context"] if hit else ""})
    return {"rows": rows, "found": found, "total": len(injected),
            "rate": round(found / float(len(injected)), 4), "source": name}


def check_ocr_route(report, cfg, n=3):
    """扫描件 OCR 路由实测：确认扫描型文档经工作流路由后能拿到文本。"""
    scans = [d for d in report["classification"]["documents"]
             if d["primary_label"] == "Scan_PDF"][:n]              # 取前若干扫描件
    rows = []                                                      # 实测明细
    for d in scans:                                                # 逐个走 OCR 解析器
        try:
            res = parsers.get_parser("ocr_parser")(d["path"], cfg, max_pages=1)
        except Exception as exc:                                   # OCR 失败也要记录
            res = {"parser": "ocr_parser", "ok": False, "pages": 0, "chars": 0,
                   "preview": "OCR 失败：%s" % exc}
        rows.append({"name": d["name"], "parser": res.get("parser"),
                     "pages": res.get("pages"), "chars": res.get("chars", 0),
                     "preview": res.get("preview", "")})
    return {"rows": rows, "workflow": DocumentIngestionWorkflow.__name__}


def _front_matter(path):
    """从专利首页著录项中抽取申请人、申请日、申请号与文献种类，用于核查版本关系。"""
    import re                                                      # 局部导入，避免顶部堆叠
    try:
        doc = fitz.open(path)                                      # 打开
    except Exception:
        return {}
    try:
        raw = "".join(doc[i].get_text() for i in range(min(doc.page_count, 2)))
    finally:
        doc.close()
    t = re.sub(r"\s+", "", raw)                                    # 去掉换行导致的字段断裂
    app = re.search(r"\((?:71|73)\)(?:申请人|专利权人)(.*?)(?:地址|\(72\)|\(74\)|\(51\))", t)
    date = re.search(r"\(22\)申请日([0-9]{4}\.[0-9]{2}\.[0-9]{2})", t)
    no = re.search(r"\(21\)申请号([0-9.]+)", t)
    kind = ("发明申请" if "发明专利申请" in t else
            "实用新型" if "实用新型专利" in t else
            "发明专利" if "发明专利" in t else "未知")
    return {"applicant": app.group(1) if app else "",
            "apply_date": date.group(1) if date else "",
            "apply_no": no.group(1) if no else "",
            "kind": kind,
            "same_day_filing": "同样的发明创造已同日申请" in t}


def check_version_conflicts(report):
    """人工确认“待确认版本冲突列表”：用著录项里的申请人/申请日等硬信息判定候选真伪。

    注意“一案两报”不能只看申请人与申请日——同一申请人同一天可能同时递交多件
    互不相同的申请。必须再加两个条件：两件文献种类不同（发明申请↔实用新型），
    且至少一方著录项带“同样的发明创造已同日申请”标记。
    """
    rows, stats = [], {"总候选": 0, "一案两报": 0, "同日姊妹申请": 0,
                       "同申请人_不同申请日": 0, "不同申请人": 0, "著录项缺失": 0}
    for pr in report["dedup"]["near_pairs"]:                       # 逐候选对
        a = _front_matter(pr["file_a"]["path"])                    # A 的著录项
        b = _front_matter(pr["file_b"]["path"])                    # B 的著录项
        stats["总候选"] += 1
        same_app = a.get("applicant") and a["applicant"] == b.get("applicant")
        same_day = same_app and a.get("apply_date") == b.get("apply_date")
        diff_kind = a.get("kind") != b.get("kind")                 # 发明申请 ↔ 实用新型
        marked = bool(a.get("same_day_filing") or b.get("same_day_filing"))
        if not (a.get("applicant") and b.get("applicant")):        # 著录项没抽到
            verdict = "著录项缺失，无法自动判定"
            stats["著录项缺失"] += 1
        elif same_app and same_day and diff_kind and marked:
            verdict = "真实版本冲突（一案两报）：同一申请人同一申请日，且一方标注同日申请"
            stats["一案两报"] += 1
        elif same_app and same_day and diff_kind:
            verdict = "同日申请的关联件（种类不同但未见同日申请标记），需人工核验是否一案两报"
            stats["同日姊妹申请"] += 1
        elif same_app and same_day:
            verdict = "同一申请人同日申请的姊妹件（文献种类相同），需人工判断是否各自保留"
            stats["同日姊妹申请"] += 1
        elif same_app:
            verdict = "同一申请人的关联申请（申请日不同），需人工判断是否保留"
            stats["同申请人_不同申请日"] += 1
        else:
            verdict = "申请人不同，疑似误报，需人工复核"
            stats["不同申请人"] += 1
        def _side(key, fm):                                        # 候选一侧的著录项摘要
            f = pr["file_" + key]
            return {"name": f["name"], "applicant": fm.get("applicant"),
                    "apply_date": fm.get("apply_date"), "kind": fm.get("kind"),
                    "apply_no": fm.get("apply_no"), "same_day": fm.get("same_day_filing"),
                    "excerpt": f["excerpt"][:120]}
        rows.append({"similarity": pr["similarity"], "hamming": pr["hamming_distance"],
                     "a": _side("a", a), "b": _side("b", b), "verdict": verdict})
    confirmed = (stats["一案两报"] + stats["同日姊妹申请"]          # 申请人一致即视为有价值候选
                 + stats["同申请人_不同申请日"])
    stats["可用候选率"] = round(confirmed / float(max(stats["总候选"], 1)), 4)
    return rows, stats


def label_review(report):
    """分类标签合理性审查：给出每个标签的数量、占比与判定依据。"""
    reason = {"Scan_PDF": "逐页字符数低于阈值，需先 OCR 才能抽取文本",
              "Text_PDF": "逐页字符数达标，可直接抽取文本层",
              "Mixed_PDF": "部分页有文本层、部分页为扫描页，需逐页分流",
              "DOCX": "Word 文档，可直接解析",
              "Markdown": "Markdown 文档，可直接解析",
              "Text": "纯文本文档，可直接解析",
              "Spreadsheet": "表格类文档",
              "Presentation": "演示文稿",
              "Other": "其他格式或无法判定类型（含加密/损坏 PDF）"}
    return [{"label": i["label"], "count": i["count"], "ratio": i["ratio"],
             "reason": reason.get(i["label"], "—")}
            for i in report["classification"]["label_distribution"]]


def main(argv=None):
    """读取质检报告并产出验证报告。"""
    ap = argparse.ArgumentParser(description="工单十八 验证与调优")
    ap.add_argument("--report", default=os.path.join(BASE_DIR, "data", "report",
                                                     "quality_report.json"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "evaluation"))
    ap.add_argument("--sample-per-label", type=int, default=30)    # 每标签抽样数
    args = ap.parse_args(argv)

    with open(args.report, "r", encoding="utf-8") as fh:           # 读质检报告
        report = json.load(fh)
    cfg, _ = load_config()                                         # 配置
    print("[验证] ① 分层抽样交叉复检（每标签 %d 个）..." % args.sample_per_label, flush=True)
    cls_rows, cls_sum = check_classification(report, args.sample_per_label)
    print("[验证] ② 敏感信息正控制样本..." , flush=True)
    ctrl = check_control(cfg)
    print("[验证] ③ 扫描件 OCR 路由实测...", flush=True)
    ocr = check_ocr_route(report, cfg)
    print("[验证] ④ 版本冲突候选人工核查（著录项比对）...", flush=True)
    ver_rows, ver_sum = check_version_conflicts(report)
    sen_rows = []                                                  # 敏感信息列表逐条抽查（见报告第四节）
    sen_sum = {"total_items": len(report["to_review"]["sensitive_items"])}
    lab_rows = label_review(report)                                # ④ 标签合理性

    os.makedirs(args.out, exist_ok=True)                           # 输出目录
    data = {"report": report, "classification_sample": cls_rows,   # 汇总
            "classification_check": cls_sum, "sensitive_sample": sen_rows,
            "sensitive_check": sen_sum, "label_review": lab_rows,
            "control_check": ctrl, "ocr_route": ocr,
            "version_rows": ver_rows, "version_check": ver_sum,
            "context_chars": cfg.get("sensitive.context_chars", 30),
            "progress_every": cfg.get("runtime.progress_every", 100)}
    path = verify_report.to_markdown(data, os.path.join(args.out, "验证报告.md"))
    with open(os.path.join(args.out, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump({"classification_check": cls_sum, "sensitive_check": sen_sum,
                   "label_review": lab_rows, "classification_sample": cls_rows,
                   "control_check": ctrl, "ocr_route": ocr,
                   "version_check": ver_sum, "version_rows": ver_rows},
                  fh, ensure_ascii=False, indent=2)
    print("[验证] 分类交叉验证一致率 %s（%d/%d）"
          % ("%.2f%%" % (cls_sum["rate"] * 100), cls_sum["agree"], cls_sum["sample_size"]))
    print("[验证] 敏感信息正控制检出 %d/%d" % (ctrl["found"], ctrl["total"]))
    print("[验证] 报告已写入 %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
