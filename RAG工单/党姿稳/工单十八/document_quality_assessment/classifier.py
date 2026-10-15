# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-18-实现文档质量评估Skill并集成至智能体工作流
# 模块职责：文档分类标签体系。主标签（互斥）表示文档类型与解析来源，决定走哪个解析器；
#           质量标签（可多选）表示需要注意的问题（重复/敏感/过短/损坏）。

import os                                                          # 后缀处理

# 后缀 → 主标签（PDF 需要结合页面类型再细化）
_EXT_LABEL = {".docx": "DOCX", ".doc": "DOCX",                     # Word
              ".md": "Markdown", ".markdown": "Markdown",          # Markdown
              ".txt": "Text",                                      # 纯文本
              ".xlsx": "Spreadsheet", ".xls": "Spreadsheet", ".csv": "Spreadsheet",
              ".pptx": "Presentation", ".ppt": "Presentation"}
# PDF 页面类型 → 主标签
_KIND_LABEL = {"scan": "Scan_PDF", "text": "Text_PDF", "mixed": "Mixed_PDF"}


def primary_label(record):
    """给出文档的主标签。"""
    ext = (record.get("ext") or "").lower()                        # 后缀
    if ext == ".pdf":                                              # PDF 看页面类型
        if record.get("error"):                                    # 打不开就无从判断类型
            return "Other"
        return _KIND_LABEL.get(record.get("pdf_kind"), "Other")    # 页面类型映射
    return _EXT_LABEL.get(ext, "Other")                            # 其他格式按后缀


def quality_tags(record, cfg, dup_exact, dup_near):
    """给出文档的质量标签列表（可多选）。"""
    tags = []                                                      # 结果
    if record.get("error"):                                        # 解析失败
        tags.append("Corrupted")
    chars = record.get("text_chars", 0)                            # 有效字符数
    empty_th = cfg.get("classify.empty_text_chars", 10)            # “无文本”阈值
    short_th = cfg.get("classify.too_short_chars", 200)            # “过短”阈值
    if not record.get("error"):                                    # 未损坏才判长度
        if chars < empty_th:
            tags.append("Empty_Text")                              # 几乎没有可提取文本
        elif chars < short_th:
            tags.append("Too_Short")                               # 文本偏少
    if record["path"] in dup_exact:                                # MD5 完全相同
        tags.append("Duplicate_Exact")
    if record["path"] in dup_near:                                 # SimHash 高相似
        tags.append("Duplicate_Near")
    if record.get("sensitive"):                                    # 命中敏感信息
        tags.append("Sensitive_Info")
    allowed = cfg.get("classify.quality_tags", [])                 # 配置中的标签白名单
    if allowed:
        tags = [t for t in tags if t in allowed]                   # 只保留配置认可的标签
    return tags


def route_of(label, tags, cfg):
    """按标签决定该文档在工作流中路由到哪个解析器。"""
    for t in cfg.get("routing.review_first_tags", []) or []:       # 需人工优先的分支
        if t in tags:
            return "human_review"
    table = cfg.get("routing.label_to_parser", {}) or {}           # 主标签 → 解析器
    return table.get(label, "passthrough_parser")                  # 默认透传


def compute(records, cfg, dedup_result):
    """主入口：为每个文档打标签并汇总分布。返回 {documents, distribution, ...}。"""
    exact_paths, near_paths = set(), set()                         # 重复文件路径集合
    for g in dedup_result.get("exact_groups", []):                 # 精确重复组
        for f in g["files"]:
            exact_paths.add(f["path"])
    for p in dedup_result.get("near_pairs", []):                   # 近似重复候选
        near_paths.add(p["file_a"]["path"])
        near_paths.add(p["file_b"]["path"])

    docs, label_count, tag_count = [], {}, {}                      # 明细与统计
    for r in records:                                              # 逐文档
        label = primary_label(r)                                   # 主标签
        tags = quality_tags(r, cfg, exact_paths, near_paths)       # 质量标签
        label_count[label] = label_count.get(label, 0) + 1         # 主标签计数
        for t in tags:                                             # 质量标签计数
            tag_count[t] = tag_count.get(t, 0) + 1
        docs.append({"path": r["path"],                            # 路径
                     "name": r.get("name", os.path.basename(r["path"])),
                     "ext": r.get("ext"),                          # 后缀
                     "primary_label": label,                       # 主标签
                     "quality_tags": tags,                         # 质量标签
                     "route": route_of(label, tags, cfg),          # 路由目标解析器
                     "text_chars": r.get("text_chars", 0),         # 字符数
                     "page_count": r.get("page_count"),            # 页数
                     "scan_ratio": r.get("scan_ratio")})           # 扫描页占比
    total = max(len(records), 1)                                   # 防零除
    labels = [{"label": k, "count": v, "ratio": round(v / float(total), 4)}
              for k, v in sorted(label_count.items(), key=lambda kv: (-kv[1], kv[0]))]
    tags = [{"tag": k, "count": v, "ratio": round(v / float(total), 4)}
            for k, v in sorted(tag_count.items(), key=lambda kv: (-kv[1], kv[0]))]
    routes = {}                                                    # 路由分布
    for d in docs:
        routes[d["route"]] = routes.get(d["route"], 0) + 1
    return {"documents": docs, "label_distribution": labels,
            "tag_distribution": tags, "route_distribution": routes}
