# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-15-优化技术图纸与文本的跨模态检索流程
# 模块职责：把解析结果切分为可检索的文本块与图文块。文本块按句边界切分并保留重叠；
#           图纸页生成"图形块"，把图号、图内标注、说明书对应段落汇聚成一条可检索的描述。

import re                                                        # 句子切分
import config                                                    # 分块参数
from logger import get_logger                                    # 日志

logger = get_logger("rag15.chunker")

_SENT_END = re.compile(r"(?<=[。；！？])")                        # 中文句末标点作为切分点
_FIG_TAIL = r"(?:所示|显示|表示|为|中|是)"                        # 图号后的谓语（图1所示/图3表示/图2中）
_ANCHOR_TPL = r"图\s*%s\s*" + _FIG_TAIL                          # 指定图号的引用锚点模板
_CUT_RE = re.compile(r"图\s*[0-9]{1,2}")                         # 任意图号引用（用于截断到下一张图）


def split_sentences(text):
    """按中文句末标点切句，过滤空白句。"""
    parts = []                                                   # 结果
    for seg in text.split("\n"):                                 # 先按行切
        for piece in _SENT_END.split(seg):                       # 再按句末标点切
            piece = piece.strip()                                # 去空白
            if piece:                                            # 非空才保留
                parts.append(piece)
    return parts


def pack_sentences(sentences, size=None, overlap=None):
    """把句子打包成不超过 size 字符、相邻重叠 overlap 字符的块。"""
    size = size or config.CHUNK_SIZE                             # 目标长度
    overlap = overlap or config.CHUNK_OVERLAP                    # 重叠长度
    chunks, buf = [], ""                                         # 结果与当前缓冲
    for sent in sentences:                                       # 逐句累加
        if buf and len(buf) + len(sent) > size:                  # 加入会超长 → 先落块
            chunks.append(buf.strip())                           # 保存当前块
            tail = buf[-overlap:] if overlap > 0 else ""         # 取尾部作为重叠
            buf = tail + sent                                    # 新块以重叠开头
        else:
            buf += sent                                          # 继续累加
    if buf.strip():                                              # 收尾
        chunks.append(buf.strip())
    return chunks


def link_description(pages, label, window=520):
    """在说明书正文中找出描述指定图号的段落（如"图1所示为..."）。

    OCR 会在行尾插入换行，直接按行/句切分会把一段完整的图纸说明截断，
    因此这里先去掉换行拼成连续文本，再以"图N所示/显示/为"为锚点截取一个窗口，
    并在窗口内回退到最近的句末标点，保证取到完整的一段描述。
    """
    pattern = re.compile(_ANCHOR_TPL % re.escape(label[1:]))      # 该图号的引用锚点
    hits = []                                                    # 命中段落
    for p in pages:                                              # 遍历所有页
        if not p["text"].strip():                                # 空页跳过
            continue
        flat = p["text"].replace("\n", "")                       # 去掉 OCR 换行，拼成连续文本
        for m in pattern.finditer(flat):                         # 逐个锚点
            seg = flat[m.start(): m.start() + window]            # 截取窗口
            nxt = _CUT_RE.search(seg, 1)                         # 窗口内下一处图号引用
            if nxt:                                              # 说明已进入下一张图的描述
                seg = seg[:nxt.start()]
            cut = max(seg.rfind("。"), seg.rfind("；"))            # 回退到最近句末
            if cut > 0:                                          # 找到句末则截到该处
                seg = seg[:cut + 1]
            if len(seg) < 40:                                    # 过短的引用（如附图说明的目录行）丢弃
                continue
            hits.append(f"(第{p['page_no']}页) {seg}")            # 带页码便于溯源
    return "\n".join(hits)


# 专利首页著录项：发明人/专利权人/发明名称/摘要等，字段极短但信息密度最高，
# 一旦被固定长度切分切断就会导致"发明人是谁"这类问题检索不到答案（信息丢失）。
# 这里按字段整体抽取，单独成块。
_BIBLIO_FIELDS = [
    ("发明人", r"\[?72[\]］]?\s*发明人\s*([^\n【\[]+)"),
    ("专利权人", r"\[?73[\]］]?\s*专利权人\s*([^\n【\[]+)"),
    ("发明名称", r"\[?54[\]］]?\s*发明名称\s*([^\n【\[]+)"),
    ("专利号", r"专利号\s*(ZL[0-9.]+)"),
    ("授权公告号", r"\[?11[\]］]?\s*授权公告号\s*(CN[0-9A-Z]+)"),
    ("申请号", r"\[?21[\]］]?\s*申请号\s*([0-9.]+)"),
    ("代理机构", r"\[?74[\]］]?\s*专利代理机构\s*([^\n【\[]+)"),
    ("摘要", r"\[?57[\]］]?\s*摘要\s*([\s\S]+?)(?=\n?\[?[0-9]{2}[\]］]|$)"),
]


def extract_biblio(pages):
    """从首页/摘要页抽取专利著录项，返回 (文本, 命中字段名列表)。"""
    found, src_pages = {}, []                                    # 字段值 / 来源页
    for p in pages:                                              # 遍历页面（著录项通常在首页）
        flat = p["text"].replace("\n", "")                       # 去换行，避免字段被 OCR 断行
        for name, pat in _BIBLIO_FIELDS:                         # 逐字段匹配
            if name in found:                                    # 已找到则跳过
                continue
            m = re.search(pat, flat)                             # 匹配
            if m:                                                # 命中
                val = re.sub(r"\s+", "", m.group(1)).strip()     # 去空白
                if val:
                    found[name] = val                            # 记录
                    src_pages.append(p["page_no"])               # 记录来源页
    if not found:                                                # 一条都没抽到
        return "", []
    lines = ["【专利著录项】" + "；".join(f"{k}：{v}" for k, v in found.items())]
    if src_pages:
        lines.append("（来源：第 %s 页）" % "、".join(str(x) for x in sorted(set(src_pages))))
    return "\n".join(lines), list(found.keys())


def build_biblio_chunk(pages, doc_name):
    """把著录项做成独立块，保证"发明人/专利权人"等字段不被切分破坏。"""
    text, fields = extract_biblio(pages)                         # 抽取
    if not text:                                                 # 未命中则不生成
        return None
    page_no = min(p["page_no"] for p in pages if p["text"].strip())  # 归属到首个非空页
    logger.info("  著录项块：命中 %d 个字段（%s）", len(fields), "、".join(fields))
    return {"chunk_id": f"{doc_name}#p{page_no}#biblio",          # 编号
            "doc": doc_name, "page_no": page_no,                  # 来源
            "type": "biblio", "source": "ocr",                    # 类型
            "labels": [], "text": text}                           # 正文


def build_text_chunks(pages, doc_name):
    """文字页 → 文本块列表。"""
    chunks = []                                                  # 结果
    for p in pages:                                              # 遍历页面
        if p["figure"] or not p["text"].strip():                 # 图纸页另作处理；空页跳过
            continue
        for i, ck in enumerate(pack_sentences(split_sentences(p["text"]))):
            chunks.append({                                      # 组装块
                "chunk_id": f"{doc_name}#p{p['page_no']}#t{i}",   # 全局唯一编号
                "doc": doc_name,                                 # 来源文档
                "page_no": p["page_no"],                         # 页码
                "type": "text",                                  # 类型
                "source": p["source"],                           # 解析来源（text/ocr）
                "labels": p["labels"],                           # 图号（文字页通常为空）
                "text": ck})                                     # 正文
    return chunks


def build_figure_chunks(pages, doc_name, figure_descs=None):
    """图纸页 → 图形块。把 VLM 描述、图内标注、说明书对应段落汇聚为一条可用文本。"""
    figure_descs = figure_descs or {}                            # 没有 VLM 描述时用空字典
    chunks = []                                                  # 结果
    for p in pages:                                              # 遍历页面
        if not p["figure"]:                                      # 只处理图纸页
            continue
        fig = p["figure"]                                        # 图纸信息
        labels = fig["labels"] or ["图1"]                        # 图号
        lines = [f"【图纸】文档 {doc_name} 第 {p['page_no']} 页，图号 {'、'.join(labels)}。"]
        for lb in labels:                                        # 逐个图号关联说明书段落
            desc = link_description(pages, lb)                   # 找"图N所示"的正文
            if desc:
                lines.append(f"说明书正文中关于{lb}的文字描述（权威依据）：\n{desc}")
        ocr_labels = [b["text"] for b in p.get("ocr_boxes", [])]  # 图内 OCR 标注文字
        if ocr_labels:
            lines.append("图内 OCR 识别到的标注文字：" + " ".join(ocr_labels))
        key = f"{doc_name}#p{p['page_no']}"                      # 描述缓存键
        if key in figure_descs:                                  # 有 VLM 生成的图形描述
            lines.append("视觉模型自动生成的图形描述（辅助信息，分辨率受限可能不准确）："
                         + figure_descs[key].strip())
        chunks.append({                                          # 组装图形块
            "chunk_id": f"{doc_name}#p{p['page_no']}#fig",        # 编号
            "doc": doc_name,                                     # 来源文档
            "page_no": p["page_no"],                             # 页码
            "type": "figure",                                    # 类型：图文块
            "source": p["source"],                               # 解析来源
            "labels": labels,                                    # 图号
            "text": "\n".join(lines)})                           # 拼接描述
        logger.info("  图纸页 p%d 生成图形块（%s，%d 字）",
                    p["page_no"], "、".join(labels), len(chunks[-1]["text"]))
    return chunks


def build_chunks(pages, doc_name, figure_descs=None):
    """汇总所有块：文本块 + 著录项块 + 图形块，按页码排序。"""
    chunks = build_text_chunks(pages, doc_name)                   # 文本块
    biblio = build_biblio_chunk(pages, doc_name)                  # 著录项块（修复字段被切断）
    if biblio:
        chunks.append(biblio)
    chunks += build_figure_chunks(pages, doc_name, figure_descs)  # 图形块
    order = {"biblio": 0, "figure": 1, "text": 2}                 # 同页内的展示顺序
    chunks.sort(key=lambda c: (c["page_no"], order.get(c["type"], 3)))
    for i, c in enumerate(chunks):                                # 补序号，便于展示
        c["order"] = i
    kinds = {t: sum(1 for c in chunks if c["type"] == t)          # 分类计数
             for t in ("text", "biblio", "figure")}
    logger.info("共生成 %d 个块（正文 %d / 著录项 %d / 图形 %d）",
                len(chunks), kinds["text"], kinds["biblio"], kinds["figure"])
    return chunks


if __name__ == "__main__":                                        # 自检
    import json                                                   # 打印样例
    from pdf_parser import parse_document                         # 复用解析器
    _pages, _meta = parse_document(config.TARGET_PDF)             # 解析
    _chunks = build_chunks(_pages, _meta["file"])                 # 分块
    for _c in _chunks:                                            # 打印概览
        print(f"[{_c['type']:>6}] {_c['chunk_id']} ({len(_c['text'])}字) "
              f"{_c['text'][:40].replace(chr(10), ' ')}")
